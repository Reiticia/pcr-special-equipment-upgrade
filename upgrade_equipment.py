"""BlueStacks 5 / PCR 特别装备安全辅助脚本（Windows）。

默认只截图和检查当前选中的装备。--execute 启用辅助执行；--fully-auto 启用全自动。
上限解锁会永久消耗基础同名装备；全自动模式还会自动确认仅消耗强化 Pt 的强化。
未知画面/无法识别/材料不足立即停止，不自动购买或分解；--scan 允许滚动查找。
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageGrab

BASE = (1416, 797)  # 实际游戏画布，不含 BlueStacks 标题栏
# 仅支持当前截图中的金色五菱形特别装备详情布局。
STAR_CENTERS = [(930, 296), (946, 296), (962, 296), (978, 296), (993, 296)]
POINTS = {"unlock": (1130, 632), "strengthen": (974, 632), "auto": (876, 708)}


class TeeOutput:
    """将 stdout/stderr 同步镜像到 UTF-8 日志；每次写入立即刷新。"""
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        if self.console is not None:
            self.console.write(text)
            self.console.flush()
        return len(text)

    def flush(self):
        self.log.flush()
        if self.console is not None:
            self.console.flush()

    def isatty(self):
        return bool(self.console is not None and self.console.isatty())

    @property
    def encoding(self):
        return "utf-8"


@contextlib.contextmanager
def mirror_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(TeeOutput(sys.stdout, log)), contextlib.redirect_stderr(TeeOutput(sys.stderr, log)):
            yield


class Stop(RuntimeError):
    pass


class MaterialsUnavailable(Stop):
    """已确认升序且当前页无更多安全材料，可取消选材后跳过，不再滚动。"""
    def __init__(self, selected, needed):
        self.selected, self.needed = selected, needed
        super().__init__(f"材料列表已确认升序，当前页无更多可用的 1 点数材料，"
                         f"仅找到 {selected}/{needed} 点安全材料，无需向下滚动")


@dataclass(frozen=True)
class Stars:
    slots: tuple[str, ...]

    @property
    def full(self):
        return all(s == "filled" for s in self.slots)

    @property
    def locked(self):
        return self.slots.count("locked")

    @property
    def level(self):
        return self.slots.count("filled")


def has_pink_diamond_outline(region: np.ndarray) -> bool:
    """检测空心粉色菱形的四条边，而非假定透明中心一定为白色。

    19x15 的局部区域；灰色/已填充状态仍由调用方优先检查。
    随机粉色纹理或单条边不能通过：四个象限都必须有轮廓证据。
    """
    if region.shape != (19, 15, 3):
        return False
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    pink = (r > 140) & (b > 120) & (r-g > 35) & (b-g > 25)
    dy, dx = np.mgrid[-9:10, -7:8]
    distance = np.abs(dx)/7 + np.abs(dy)/9
    ring = (distance >= .70) & (distance <= 1.30)
    interior = distance < .45
    if np.mean(pink[ring]) < .25 or np.mean(pink[interior]) > .20:
        return False
    return all(np.count_nonzero(pink & ring & (sx*dx > 0) & (sy*dy > 0)) >= 3
               for sx in (-1, 1) for sy in (-1, 1))


def has_solid_gray_fill(region: np.ndarray) -> bool:
    """未强化菱形必须是均匀灰色实心，不能仅凭中心像素判定。

    锁定菱形透出的图案可能恰好呈灰色；检查整个内部区域的覆盖率和波动。
    粉边的已解锁菱形内部略带紫色，沿用原灰色容差，不把它当成锁定。
    """
    if region.shape != (19, 15, 3):
        return False
    dy, dx = np.mgrid[-9:10, -7:8]
    pixels = region[(np.abs(dx)/7 + np.abs(dy)/9) < .55]
    gray = ((pixels.mean(1) > 65) & (pixels.mean(1) < 195)
            & (pixels.max(1)-pixels.min(1) < 45))
    return bool(np.mean(gray) >= .85 and np.max(pixels.std(0)) <= 20)


def classify_stars(image: np.ndarray, centers=None) -> Stars:
    """RGB 图像。模糊或非五菱形布局不猜测，抛出 Stop。"""
    slots = []
    for x, y in (STAR_CENTERS if centers is None else centers):
        patch = image[y-2:y+3, x-1:x+2].astype(float)
        r, g, b = np.median(patch, axis=(0, 1))
        # 图标内填充色，不能仅凭“当前强化条已满”判断最高上限。
        cyan = b > 175 and g > 130 and r < 155 and b-r > 65
        pink = r > 170 and b > 155 and g < 185 and min(r, b)-g > 45
        gray = 65 < (r+g+b)/3 < 195 and max(r, g, b)-min(r, g, b) < 45
        region = image[y-9:y+10, x-7:x+8].astype(float)
        if cyan or pink:
            slots.append("filled")
        elif gray and has_solid_gray_fill(region):
            slots.append("empty")
        elif has_pink_diamond_outline(region):
            slots.append("locked")
        else:
            raise Stop(f"无法可靠识别菱形 ({x}, {y})，RGB={r:.0f},{g:.0f},{b:.0f}；不执行。")
    # 星级必须从左到右连续；当前版本最多两个可解锁的粉色菱形。
    rank = {"filled": 0, "empty": 1, "locked": 2}
    if any(rank[a] > rank[b] for a, b in zip(slots, slots[1:])) or slots[:3].count("locked"):
        raise Stop(f"星级排列不符合已验证的五菱形布局：{slots}")
    return Stars(tuple(slots))


def artwork(image):
    # 排除底部星级、右上角色头像以及保护按钮。
    return image[222:277, 924:997].astype(float)


def visible_candidates(image):
    """发现列表中的菱形行；只返回能识别全部五个菱形的未满星格子。

    候选检测不授权消耗；点击后必须用右侧详情再次验证。
    """
    a = image[155:675, 50:850].astype(float)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    gray = (a.max(2)-a.min(2) < 30) & (a.mean(2) > 120) & (a.mean(2) < 180)
    cyan = (b > 175) & (g > 130) & (r < 155) & (b-r > 65)
    mask = ((gray | cyan).astype(np.uint8)*255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (8 <= w <= 19 and 12 <= h <= 25 and 45 <= area <= 230):
            continue
        cx, cy = x+50+(w-1)/2, y+155+(h-1)/2
        expected = [94+159.3*c+16.2*s for c in range(5) for s in range(3)]
        if min(abs(cx-ex) for ex in expected) < 3 and 220 <= cy <= 663:
            if not any(abs(cy-old) < 4 for old in rows):
                rows.append(cy)
    candidates = []
    for y in sorted(rows):
        for column in range(5):
            centers = [(round(94+159.3*column+16.2*s), round(y)) for s in range(5)]
            try:
                stars = classify_stars(image, centers)
            except Stop:
                continue
            if not stars.full:
                candidates.append((round(142+159.3*column), round(y-55), stars))
    return candidates


def material_sort_direction(words):
    """仅识别解锁材料列表顶部的排序按钮，不匹配背包或其他页面文字。"""
    directions = set()
    for box, text, score in words:
        x, y = np.mean(box, axis=0)
        if score >= .85 and 400 <= x <= 545 and 140 <= y <= 195 and text in ("升序", "降序"):
            directions.add(text)
    if len(directions) > 1:
        raise Stop("解锁材料排序按钮识别冲突，不点击。")
    return next(iter(directions), None)


def unlock_progress(words):
    values = []
    for box, text, score in words:
        x, y = np.mean(box, axis=0)
        match = re.fullmatch(r"(\d+)/(\d+)[.。]?", text)
        if score >= .85 and 1130 <= x <= 1330 and 285 <= y <= 365 and match:
            values.append(tuple(map(int, match.groups())))
    if len(values) != 1 or not 0 <= values[0][0] <= values[0][1] <= 20 or values[0][1] == 0:
        raise Stop("无法唯一识别上限解锁 Pt 的当前值/需求值；不选择材料。")
    return values[0]


def material_candidates(image, words):
    """只接受基础三灰+两锁，且点数标签不是 3 点等高级材料。

    游戏上限解锁候选页提供同名约束；不用于普通强化页。
    """
    a = image[200:645, 50:682].astype(float)
    mask = (((a.max(2)-a.min(2) < 30) & (a.mean(2) > 120)
             & (a.mean(2) < 180)).astype(np.uint8)*255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 8 <= w <= 20 and 12 <= h <= 26 and 45 <= cv2.contourArea(contour) <= 250:
            cy = y+200+(h-1)/2
            if 325 <= cy <= 633 and not any(abs(cy-old) < 4 for old in rows):
                rows.append(cy)
    result = []
    for y in sorted(rows):
        for col in range(4):
            left = 73+155*col
            centers = [(round(92.5+155*col+17.3*s), round(y)) for s in range(5)]
            try:
                stars = classify_stars(image, centers)
            except Stop:
                continue
            if stars.slots != ("empty", "empty", "empty", "locked", "locked"):
                continue
            labels = []
            for box, text, score in words:
                tx, ty = np.mean(box, axis=0)
                if left-5 <= tx <= left+88 and y-122 <= ty <= y-85 and score >= .85:
                    labels.append(text)
            # OCR 会漏掉贴边的数字 1；五菱形状态及点击后 +1 Pt 额外核验。
            if not any(re.fullmatch(r"1?点数", text) for text in labels):
                continue
            # 右上角须主要为金色背景；头像/明显锁定或选择标记不视为安全材料。
            top = round(y-121)
            badge = image[top+5:top+34, left+105:left+136].astype(float)
            r, g, b = badge[:, :, 0], badge[:, :, 1], badge[:, :, 2]
            if np.mean((r > 195) & (g > 150) & (b < 220) & (r-b > 35)) < .65:
                continue
            result.append((left+70, round(y-55)))
    return result


def validate_pt_confirmation(image, words):
    """无人值守确认仅允许强化 Pt，绝不确认吞装备或购买。"""
    if not Assistant.has(words, "特别装备强化确认", (400, 20, 1020, 110)):
        raise Stop("不是已验证的强化确认窗口。")
    for forbidden in ("不足", "购买", "宝石", "充值"):
        if Assistant.has(words, forbidden, (360, 30, 1060, 750)):
            raise Stop(f"强化确认出现 {forbidden}，停止。")
    template = np.asarray(Image.open(Path(__file__).parent / "assets/strengthen-pt.png").convert("RGB")).astype(float)
    icon = image[269:335, 394:474].astype(float)
    if float(np.mean(np.abs(icon-template))) > 12:
        raise Stop("消耗道具不是已验证的强化 Pt 图标；不自动确认吞装备。")
    # Pt 之外的整个消耗列表必须为空，含其他装备/滚动条都拒绝。
    for region in (image[260:575, 490:1035], image[367:575, 382:490]):
        if np.mean(region.min(2) < 220) > .008:
            raise Stop("消耗列表还有其他内容，不能保证仅消耗强化 Pt。")
    def number(region, pattern):
        matches = []
        for box, text, score in words:
            x, y = np.mean(box, axis=0)
            m = re.fullmatch(pattern, text.replace(",", ""))
            if score >= .85 and region[0] <= x <= region[2] and region[1] <= y <= region[3] and m:
                matches.append(int(m.group(1)))
        if len(matches) != 1:
            raise Stop("无法唯一读取消耗数量，不自动确认。")
        return matches[0]
    pt = number((390, 332, 486, 365), r"[×xX](\d+)")
    mana = number((575, 182, 709, 226), r"(\d+)")
    owned = number((870, 182, 1035, 226), r"(\d+)")
    if not 0 < pt <= 6000 or not 0 <= mana <= min(owned, 300000):
        raise Stop(f"单件强化预算异常：Pt={pt}, 玛那={mana}, 持有={owned}。")
    return pt, mana


def scrollbar_bottom(image):
    bar = image[165:668, 860:871].astype(float)
    blue = (bar[:, :, 2] > 150) & (bar[:, :, 2]-bar[:, :, 0] > 45) & (bar[:, :, 1] > 110)
    ys, _ = np.where(blue)
    return bool(len(ys) and ys.max()+165 >= 654)


class Window:
    def __init__(self, title):
        if sys.platform != "win32":
            raise Stop("只支持 Windows。")
        # 必须在 Win32/Pillow 获取坐标前设置 DPI，兼容 150% 缩放。
        ctypes.windll.user32.SetProcessDPIAware()
        import win32gui
        import win32con
        self.gui, self.con = win32gui, win32con
        matches = []
        win32gui.EnumWindows(lambda h, _: matches.append(h) if
                             win32gui.IsWindowVisible(h) and
                             win32gui.GetWindowText(h) == title else None, None)
        if len(matches) != 1:
            raise Stop(f"精确标题 {title!r} 找到 {len(matches)} 个窗口；请使用 --title 指定。")
        self.hwnd = matches[0]
        children = []
        win32gui.EnumChildWindows(self.hwnd, lambda h, _: children.append(h) if
                                  win32gui.GetWindowText(h) == "HD-Player" else None, None)
        if len(children) != 1:
            raise Stop("无法唯一定位 HD-Player 游戏画布。")
        self.canvas = children[0]
        self.rect = self.bounds()
        w, h = self.rect[2]-self.rect[0], self.rect[3]-self.rect[1]
        if abs(w/h-16/9) > .025 or w < 1000:
            raise Stop("请将游戏画布设为横屏 16:9，且宽度至少 1000 像素。")

    def bounds(self):
        return self.gui.GetWindowRect(self.canvas)

    def focus(self):
        if self.gui.IsIconic(self.hwnd):
            self.gui.ShowWindow(self.hwnd, self.con.SW_RESTORE)
        if self.gui.GetForegroundWindow() != self.hwnd:
            try:
                self.gui.SetForegroundWindow(self.hwnd)
            except Exception:
                # Windows 前台锁可能阻止从终端直接激活窗口；安全失败，不盲点。
                pass
        time.sleep(.5)
        if self.gui.GetForegroundWindow() != self.hwnd:
            raise Stop("无法激活 BlueStacks，请重新运行后立即手动点其标题栏。")

    def guard(self):
        if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
            raise Stop("检测到 Esc，已停止。")
        if self.bounds() != self.rect:
            raise Stop("窗口位置或尺寸改变，停止以避免误点。重新运行即可。")
        if self.gui.GetForegroundWindow() != self.hwnd:
            raise Stop("BlueStacks 不在前台，停止。")
        # 鼠标移到屏幕左上角也会停止自动点击。
        x, y = self.gui.GetCursorPos()
        if x < 4 and y < 4:
            raise Stop("鼠标位于左上角，紧急停止。")

    def grab(self):
        self.guard()
        return np.asarray(ImageGrab.grab(bbox=self.rect, all_screens=True).convert("RGB")
                          .resize(BASE, Image.Resampling.LANCZOS))

    def click(self, x, y):
        self.guard()
        l, t, r, b = self.rect
        import win32api
        win32api.SetCursorPos((round(l+x*(r-l)/BASE[0]), round(t+y*(b-t)/BASE[1])))
        win32api.mouse_event(self.con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        win32api.mouse_event(self.con.MOUSEEVENTF_LEFTUP, 0, 0)
        time.sleep(.9)

    def scroll_down(self):
        self.guard()
        import win32api
        l, t, r, b = self.rect
        win32api.SetCursorPos((round(l+470*(r-l)/BASE[0]), round(t+430*(b-t)/BASE[1])))
        # 一格滚轮，小幅重叠滚动；不直接拖到底，以免遗漏中间装备。
        win32api.mouse_event(self.con.MOUSEEVENTF_WHEEL, 0, 0, -120, 0)
        time.sleep(1)


class Assistant:
    def __init__(self, args):
        from rapidocr_onnxruntime import RapidOCR
        self.args = args
        self.win = Window(args.title)
        self.ocr = RapidOCR()
        self.directory = getattr(args, "run_dir", Path(args.output) / time.strftime("%Y%m%d-%H%M%S"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.material_shortages = []  # 本次运行内记住同名、同图案、同等/更高需求的缺料情况。
        self.current_name = ""
        print("3 秒后检查 BlueStacks；若自动激活失败，请在这段时间手动点其标题栏。")
        time.sleep(3)
        self.win.focus()

    def snapshot(self, label):
        image = self.win.grab()
        self.sequence += 1
        name = f"{self.sequence:03d}-{label}"
        Image.fromarray(image).save(self.directory / f"{name}.png")
        result, _ = self.ocr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        words = [(box, text.replace(" ", ""), float(score))
                 for box, text, score in (result or []) if score >= .70]
        (self.directory / f"{name}.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
        return image, words

    @staticmethod
    def has(words, text, region=None):
        for box, value, _ in words:
            x, y = np.mean(box, axis=0)
            if text in value and (region is None or
                                 region[0] <= x <= region[2] and region[1] <= y <= region[3]):
                return True
        return False

    def require(self, words, text, region=None):
        if not self.has(words, text, region):
            raise Stop(f"未识别到预期页面文字：{text}。截图已保存。")

    def detail(self):
        image, words = self.snapshot("detail")
        self.require(words, "道具一览", (0, 0, 350, 110))
        self.require(words, "道具详情", (1000, 165, 1280, 240))
        self.require(words, "上限解锁", (1020, 595, 1230, 680))
        self.require(words, "强化Pt", (900, 520, 1040, 580))
        stars = classify_stars(image)
        title_words = [(np.mean(box, axis=0)[1], text) for box, text, score in words
                       if score >= .85 and 955 <= np.mean(box, axis=0)[0] <= 1290
                       and 110 <= np.mean(box, axis=0)[1] <= 178]
        self.current_name = "".join(text for _, text in sorted(title_words))
        print(f"当前选中装备：{stars.level}/5 已强化，{stars.locked} 个上限待解锁。")
        return image, stars

    def wait_screen(self, title):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            image, words = self.snapshot("wait")
            if self.has(words, title, (250, 15, 1150, 115)):
                return image, words
            time.sleep(.8)
        raise Stop(f"等待页面 {title} 超时，不再点击。")

    def open(self, button, title):
        self.detail()  # 点击前再次验证处于装备详情页，而非其他弹窗。
        self.win.click(*POINTS[button])
        return self.wait_screen(title)

    def resume(self, instruction):
        print("\n" + instruction)
        answer = input("完成后回到终端输入 YES 继续；其他输入停止：").strip()
        if answer != "YES":
            raise Stop("用户停止。")
        print("3 秒后重新激活 BlueStacks；请不要移动窗口。")
        time.sleep(3)
        self.win.focus()

    @staticmethod
    def same_item(before, after):
        delta = np.mean(np.abs(artwork(before)-artwork(after)))
        if delta > 10:
            raise Stop("选中装备图像发生变化，无法保证仍是同一件；停止。")

    def known_material_shortage(self, image, stars):
        if not stars.locked or not self.current_name:
            return False  # 已解锁到最高上限的装备仍可直接强化。
        return any(rule[0] == self.current_name and stars.locked >= rule[1]
                   and np.mean(np.abs(artwork(image)-rule[2])) < 2
                   for rule in self.material_shortages)

    def skip_material_shortage(self, original, stars, reason):
        # 只点击关闭，取消整个材料选择，不触碰解锁提交按钮。
        _, words = self.snapshot("cancel-shortage")
        self.require(words, "特别装备上限解锁", (250, 15, 1150, 115))
        self.require(words, "关闭", (400, 675, 690, 750))
        self.win.click(545, 708)
        image, after = self.wait_detail()
        self.same_item(original, image)
        if after != stars:
            raise Stop("取消选材后星级状态变化，不能安全跳过；停止。")
        if not self.current_name:
            raise Stop("已取消材料选择，但无法识别装备名称用于防重试；停止。")
        self.material_shortages.append((self.current_name, stars.locked, artwork(image).copy()))
        print(f"跳过 {self.current_name}：{reason}；已取消选材，未消耗资源，继续查找下一件。", flush=True)

    def search(self):
        """当前屏没有候选才滚动；到底或滚动受阻必须区分。"""
        previous = None
        still = 0
        rejected = set()
        for page in range(self.args.max_scrolls+1):
            image, words = self.snapshot("scan")
            self.require(words, "道具一览", (0, 0, 350, 110))
            self.require(words, "道具详情", (1000, 165, 1280, 240))
            self.require(words, "上限解锁", (1020, 595, 1230, 680))
            for x, y, estimate in visible_candidates(image):
                if (x, y) in rejected:
                    continue
                print(f"发现候选 ({x},{y})：{estimate.level}/5，点击核对详情。")
                self.win.click(x, y)
                try:
                    detail_image, stars = self.detail()
                except Stop as exc:
                    # 星级/页面识别不确定不能继续盲目滚动。
                    raise Stop(f"候选详情核对失败：{exc}") from exc
                if not stars.full:
                    if self.known_material_shortage(detail_image, stars):
                        print(f"跳过 {self.current_name}：本次已确认同类材料不足，且解锁需求未降低。", flush=True)
                    else:
                        return True
                rejected.add((x, y))
            current = image[180:665, 80:840].astype(float)
            difference = float(np.mean(np.abs(current-previous))) if previous is not None else 255
            still = still+1 if difference < 1.0 else 0
            if still >= 2 and scrollbar_bottom(image):
                print("已确认列表到底，当前起点至底部未找到可可靠识别的未满星装备。")
                return False
            if still >= 4:
                raise Stop("滚动后列表未变化，但未确认到底；可能被遮挡/滚动失效，不误报找不到。")
            if page == self.args.max_scrolls:
                raise Stop("达到滚动安全上限，未确认搜索完毕；可提高 --max-scrolls 后继续。")
            print(f"当前可见区域无候选，向下滚动（{page+1}/{self.args.max_scrolls}）。")
            previous = current
            self.win.scroll_down()
            rejected.clear()
        return False

    def ensure_materials_ascending(self, image, words):
        """升序不点；降序只切换一次，并等待看到升序后才允许选材。"""
        self.require(words, "特别装备上限解锁", (250, 15, 1150, 115))
        direction = material_sort_direction(words)
        if direction == "升序":
            print("解锁材料已是升序，跳过排序切换。", flush=True)
            return image, words
        if direction != "降序":
            raise Stop("未识别到解锁材料排序按钮的升序/降序状态，不盲点。")
        print("解锁材料为降序，点击一次切换升序，优先查找 1 点数材料。", flush=True)
        self.win.click(470, 168)
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            image, words = self.snapshot("materials-sort-ascending")
            self.require(words, "特别装备上限解锁", (250, 15, 1150, 115))
            if material_sort_direction(words) == "升序":
                print("已确认解锁材料排序为升序，重新识别当前材料列表。", flush=True)
                return image, words
            time.sleep(.4)
        raise Stop("点击排序后未确认变为升序，不重复点击，也不继续选材。")

    def select_materials(self):
        """仅在已确认升序的当前页选材，不滚动；每次核对 +1 Pt，满额即停。"""
        image, words = self.snapshot("materials-start")
        self.require(words, "特别装备上限解锁", (250, 15, 1150, 115))
        current, needed = unlock_progress(words)
        if current:
            raise Stop("解锁页已有材料选择，无法证明其来源，请先清空后重启。")
        image, words = self.ensure_materials_ascending(image, words)
        if unlock_progress(words) != (current, needed):
            raise Stop("排序后材料 Pt 或需求发生变化，停止，不继续选材。")
        chosen = set()
        while current < needed:
            self.require(words, "特别装备上限解锁", (250, 15, 1150, 115))
            if material_sort_direction(words) != "升序":
                raise Stop("选材时未能确认保持升序，不按缺料跳过，也不继续点击。")
            candidates = [p for p in material_candidates(image, words) if p not in chosen]
            if not candidates:
                # 升序下后续材料点数只会更高，无需滚到列表底部才判定缺料。
                raise MaterialsUnavailable(current, needed)
            x, y = candidates[0]
            print(f"自动选择基础未强化同名材料 ({x},{y})；当前 Pt={current}/{needed}。", flush=True)
            before = current
            self.win.click(x, y)
            image, words = self.snapshot("material-selected")
            self.require(words, "特别装备上限解锁", (250, 15, 1150, 115))
            current, denominator = unlock_progress(words)
            if denominator != needed or current != before+1:
                raise Stop("选择后 Pt 未恰好增加 1，停止；请检查已选材料（尚未提交消耗）。")
            chosen.add((x, y))
        print(f"已自动选好 {current} 点基础材料，满足本次解锁；尚未提交消耗。", flush=True)
        return current

    def wait_detail(self):
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            image, words = self.snapshot("return-detail")
            if self.has(words, "道具一览", (0, 0, 350, 110)) and self.has(words, "道具详情", (1000, 165, 1280, 240)):
                return image, classify_stars(image)
            for forbidden in ("不足", "购买", "充值"):
                if self.has(words, forbidden):
                    raise Stop(f"操作后出现 {forbidden}，停止。")
            time.sleep(.8)
        raise Stop("操作后未返回装备详情；不重复提交，请检查截图。")

    def confirm_unlock(self):
        _, words = self.snapshot("before-unlock-submit")
        self.require(words, "特别装备上限解锁", (250, 15, 1150, 115))
        current, needed = unlock_progress(words)
        if current != needed:
            raise Stop("材料 Pt 未达到需求，不解锁。")
        self.require(words, "解锁上限", (740, 675, 1000, 750))
        print(f"自动提交上限解锁：{current}/{needed} 点基础材料。", flush=True)
        self.win.click(868, 708)
        self.wait_detail()

    def auto_strengthen(self):
        _, words = self.snapshot("before-auto-strengthen")
        self.require(words, "特别装备强化", (250, 15, 1150, 115))
        self.require(words, "自动强化", (725, 665, 1025, 760))
        self.win.click(*POINTS["auto"])
        image, words = self.wait_screen("特别装备强化确认")
        pt, mana = validate_pt_confirmation(image, words)
        self.require(words, "确认", (750, 670, 1020, 750))
        print(f"自动确认强化：仅消耗 {pt} 强化 Pt、{mana} 玛那，不吞其他装备。", flush=True)
        self.win.click(868, 708)
        return self.wait_detail()

    def process(self):
        original, stars = self.detail()
        if stars.full:
            print("已达最高五星，跳过，不打开强化或解锁。")
            return
        if not self.args.execute:
            print("预览模式：未点击升级按钮、未消耗资源。使用 --execute 启动安全辅助流程。")
            return
        for _ in range(2):
            if not stars.locked:
                break
            self.open("unlock", "特别装备上限解锁")
            if self.args.auto_materials:
                try:
                    count = self.select_materials()
                except MaterialsUnavailable as exc:
                    self.skip_material_shortage(original, stars, exc)
                    return
                if self.args.fully_auto:
                    self.confirm_unlock()
                else:
                    self.resume(
                        f"【材料已自动选择】已选 {count} 点基础未强化同名装备。\n"
                        "请核对已选材料，然后在游戏中点击解锁上限并完成确认。\n"
                        "完成后回到这件装备的详情页，再输入 YES；不要切换选中装备。")
            else:
                self.resume(
                    "【人工材料关卡】请在游戏中只选愿意消耗的同名装备。\n"
                    "不要选择已强化、已装备、受保护或准备保留的装备。\n"
                    "材料不足就停止；不要购买。请手动完成一次上限解锁、关闭结果弹窗，\n"
                    "回到这件装备的详情页。不要更换选中装备或滚动列表。\n"
                    "脚本不会代替你挑选、吞掉装备或点击解锁确认。")
            image, next_stars = self.detail()
            self.same_item(original, image)
            if next_stars.locked >= stars.locked:
                raise Stop("未检测到上限提高，可能取消或材料不足；停止。")
            stars = next_stars
        if stars.locked:
            raise Stop("仍有未解锁上限，停止。")
        if stars.full:
            print("已达到最高五星。")
            return
        self.open("strengthen", "特别装备强化")
        if self.args.fully_auto:
            image, stars = self.auto_strengthen()
            self.same_item(original, image)
            if not stars.full:
                raise Stop("强化后未达到最高五星，停止，不反复消耗资源。")
            print("验证成功：该装备已达到最高五星，继续下一件。", flush=True)
            return
        self.resume(
            "【强化材料关卡】请先核对游戏的“自动强化设定”。\n"
            "“优先消耗强化 Pt”不等于“只消耗 Pt”：Pt 不足时可能吞装备。\n"
            "只有确认设置及现有材料不会消耗需保留的装备，才输入 YES。\n"
            "请关闭设置，停留在“特别装备强化”页面，不要先选材料。\n"
            "下一步脚本会点击一次“自动强化”；该按钮可能直接消耗资源。")
        _, words = self.snapshot("before-auto")
        self.require(words, "特别装备强化", (250, 15, 1150, 115))
        self.require(words, "自动强化", (725, 665, 1025, 760))
        self.win.click(*POINTS["auto"])
        self.snapshot("after-auto")
        self.resume(
            "已点击一次自动强化。请核对游戏实际选中的材料/数量和消耗，\n"
            "仅在正确时手动完成后续确认；出现材料不足、购买或意外弹窗就停止。\n"
            "完成后关闭弹窗，回到同一件装备详情页，再输入 YES 验证。")
        image, stars = self.detail()
        self.same_item(original, image)
        if not stars.full:
            raise Stop("未验证到最高五星，不报告成功，也不反复消耗材料。")
        print("验证成功：该装备已达到最高五星。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="允许点击；消耗材料仍需人工关卡")
    parser.add_argument("--repeat", action="store_true", help="完成后手动选择下一件，重复运行")
    parser.add_argument("--fully-auto", action="store_true", help="启用查找、基础材料选择、解锁提交、仅 Pt 强化确认，不再输入 YES")
    parser.add_argument("--auto-materials", action="store_true", help="解锁时自动选择基础未强化同名装备，最终消耗仍人工确认")
    parser.add_argument("--scan", action="store_true", help="自动查找并选择候选；找不到时向下滚动（预览也会导航）")
    parser.add_argument("--max-scrolls", type=int, default=2000, help="单次搜索最多滚动次数，达到只停止、不声称搜完")
    parser.add_argument("--max-items", type=int, default=100, help="单次执行最多处理装备数")
    parser.add_argument("--title", default="BlueStacks 5", help="精确窗口标题")
    parser.add_argument("--output", default="runs", help="截图和 OCR 日志目录")
    args = parser.parse_args()
    if args.fully_auto:
        args.execute = args.auto_materials = args.scan = True
    if args.max_scrolls < 1 or args.max_items < 1:
        parser.error("--max-scrolls 和 --max-items 必须为正整数")
    args.run_dir = Path(args.output) / (time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    log_path = args.run_dir / "run.log"
    with mirror_log(log_path):
        print(f"日志文件：{log_path.resolve()}")
        print(f"开始：{time.strftime('%Y-%m-%d %H:%M:%S')}，PID={os.getpid()}，"
              f"全自动={args.fully_auto}，执行={args.execute}，查找={args.scan}")
        code = run_session(args)
        print(f"结束：{time.strftime('%Y-%m-%d %H:%M:%S')}，退出码={code}")
        return code


def run_session(args):
    try:
        bot = Assistant(args)
        print(f"诊断截图：{bot.directory}")
        for _ in range(args.max_items):
            if args.scan and not bot.search():
                break
            bot.process()
            if args.scan:
                if not args.execute:
                    break  # 预览仅定位第一件，不升级也不把它当成已处理。
                continue
            if not args.repeat:
                break
            bot.resume("请选择下一件装备并保持详情页。已满五星会自动跳过。")
        else:
            print("已达到 --max-items 安全上限；未声称整个背包已完成。")
    except (Stop, KeyboardInterrupt) as exc:
        print(f"\n已停止：{exc}", file=sys.stderr)
        return 1
    except Exception:
        print("\n未预期异常，已停止自动操作；完整回溯如下：", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
