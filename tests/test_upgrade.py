import unittest
import json
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from upgrade_equipment import (
    BASE, STAR_CENTERS, Assistant, Stop, classify_stars,
    scrollbar_bottom, visible_candidates, material_candidates, unlock_progress, validate_pt_confirmation,
    has_pink_diamond_outline, MaterialsUnavailable, Stars, artwork,
    has_solid_gray_fill, mirror_log, main,
)

FIXTURES = Path(__file__).parent / 'fixtures'


def frame(name, x=914, y=190):
    image = np.full((BASE[1], BASE[0], 3), 255, dtype=np.uint8)
    crop = np.array(Image.open(FIXTURES / name).convert('RGB'))
    h, w = crop.shape[:2]
    image[y:y+h, x:x+w] = crop
    return image


class RecognitionTests(unittest.TestCase):
    def shortage_bot(self):
        bot = Assistant.__new__(Assistant)
        bot.args = SimpleNamespace(max_scrolls=8, execute=True, auto_materials=True, fully_auto=True)
        bot.current_name = '贤王的耳饰'
        bot.material_shortages = []
        bot.win = MagicMock()
        bot.require = MagicMock()
        return bot

    def test_real_material_bottom_raises_recoverable_shortage(self):
        bot = self.shortage_bot()
        image = frame('materials-exhausted.png', 50, 200)
        words = json.loads((FIXTURES / 'materials-exhausted-ocr.json').read_text(encoding='utf-8'))
        bot.snapshot = MagicMock(return_value=(image, words))
        with self.assertRaises(MaterialsUnavailable) as caught:
            bot.select_materials()
        self.assertEqual((caught.exception.selected, caught.exception.needed), (0, 2))
        self.assertEqual(bot.win.scroll_down.call_count, 2)
        bot.win.click.assert_not_called()

    def test_shortage_cancels_and_never_submits(self):
        bot = self.shortage_bot()
        image = frame('detail-sage-empty.png')
        stars = classify_stars(image)
        bot.detail = MagicMock(return_value=(image, stars))
        bot.open = MagicMock()
        bot.select_materials = MagicMock(side_effect=MaterialsUnavailable(1, 2))
        bot.snapshot = MagicMock(return_value=(image, []))
        bot.wait_detail = MagicMock(return_value=(image, stars))
        bot.confirm_unlock = MagicMock()
        bot.auto_strengthen = MagicMock()
        bot.process()
        bot.win.click.assert_called_once_with(545, 708)
        bot.confirm_unlock.assert_not_called()
        bot.auto_strengthen.assert_not_called()
        self.assertTrue(bot.known_material_shortage(image, stars))
        self.assertFalse(bot.known_material_shortage(image, Stars(('empty',)*5)))
        self.assertFalse(bot.known_material_shortage(image, Stars(('empty',)*4+('locked',))))
        bot.current_name = '另一种耳饰'
        self.assertFalse(bot.known_material_shortage(image, stars))

    def test_material_scroll_failure_still_stops(self):
        bot = self.shortage_bot()
        image = frame('detail-sage-empty.png')
        bot.detail = MagicMock(return_value=(image, classify_stars(image)))
        bot.open = MagicMock()
        bot.select_materials = MagicMock(side_effect=Stop('滚动失效，未确认到底'))
        bot.skip_material_shortage = MagicMock()
        with self.assertRaisesRegex(Stop, '滚动失效'):
            bot.process()
        bot.skip_material_shortage.assert_not_called()

    def test_search_moves_past_known_shortage(self):
        bot = self.shortage_bot()
        image = frame('detail-sage-empty.png')
        stars = classify_stars(image)
        bot.material_shortages = [('贤王的耳饰', 2, artwork(image))]
        bot.snapshot = MagicMock(return_value=(image, []))
        def details():
            bot.current_name = '贤王的耳饰' if bot.win.click.call_count == 1 else '另一种耳饰'
            return image, stars
        bot.detail = details
        with patch('upgrade_equipment.visible_candidates', return_value=[(142, 446, stars), (301, 446, stars)]):
            self.assertTrue(bot.search())
        self.assertEqual(bot.win.click.call_count, 2)
        self.assertEqual(bot.current_name, '另一种耳饰')

    def test_pt_only_confirmation(self):
        image = frame('pt-confirm.png', 360, 30)
        words = json.loads((FIXTURES / 'pt-confirm-ocr.json').read_text(encoding='utf-8'))
        self.assertEqual(validate_pt_confirmation(image, words), (6000, 180000))

    def test_extra_equipment_never_confirmed(self):
        image = frame('pt-confirm.png', 360, 30)
        image[270:350, 510:590] = (80, 170, 250)
        words = json.loads((FIXTURES / 'pt-confirm-ocr.json').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(Stop, '其他内容'):
            validate_pt_confirmation(image, words)

    def test_wrong_pt_icon_never_confirmed(self):
        image = frame('pt-confirm.png', 360, 30)
        image[269:335, 394:474] = 0
        words = json.loads((FIXTURES / 'pt-confirm-ocr.json').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(Stop, '图标'):
            validate_pt_confirmation(image, words)

    def test_high_cost_never_confirmed(self):
        image = frame('pt-confirm.png', 360, 30)
        words = json.loads((FIXTURES / 'pt-confirm-ocr.json').read_text(encoding='utf-8'))
        words = [(b, '×99999' if t == '×6000' else t, s) for b, t, s in words]
        with self.assertRaisesRegex(Stop, '预算'):
            validate_pt_confirmation(image, words)

    def test_materials_skip_full_stars(self):
        image = frame('materials.png', 50, 200)
        words = json.loads((FIXTURES / 'materials-ocr.json').read_text(encoding='utf-8'))
        points = material_candidates(image, words)
        self.assertGreaterEqual(len(points), 2)
        self.assertTrue(all(y > 400 for x, y in points))
        self.assertEqual(unlock_progress(words), (0, 2))

    def test_materials_reject_three_point_label(self):
        image = frame('materials.png', 50, 200)
        words = json.loads((FIXTURES / 'materials-ocr.json').read_text(encoding='utf-8'))
        words = [(box, '3点数' if '点数' in text else text, score) for box, text, score in words]
        self.assertEqual(material_candidates(image, words), [])

    def test_missing_or_ambiguous_progress_stops(self):
        with self.assertRaises(Stop):
            unlock_progress([])
        words = json.loads((FIXTURES / 'materials-ocr.json').read_text(encoding='utf-8'))
        with self.assertRaises(Stop):
            unlock_progress(words+words)

    def test_sage_earring_transparent_locked_star_regression(self):
        image = frame('detail-sage-empty.png')
        # 用户报错现场：透明菱形中心是装备的棕色图案，不是白色。
        np.testing.assert_array_equal(np.median(image[294:299, 992:995], axis=(0, 1)), [221, 194, 173])
        self.assertEqual(classify_stars(image).slots,
                         ('empty', 'empty', 'empty', 'locked', 'locked'))

    def test_sage_grid_recognizes_new_type_without_full_row(self):
        candidates = visible_candidates(frame('grid-sage.png', 50, 155))
        self.assertEqual(len(candidates), 12)
        self.assertIn((142, 446), [(x, y) for x, y, _ in candidates])
        self.assertTrue(all(stars.level == 0 and stars.locked == 2 for _, _, stars in candidates))

    def test_brown_center_without_outline_fails_closed(self):
        image = frame('detail-sage-empty.png')
        image[287:306, 986:1001] = (221, 194, 173)
        with self.assertRaises(Stop):
            classify_stars(image)

    def test_broken_pink_outline_is_not_enough(self):
        image = frame('detail-sage-empty.png').astype(float)
        region = image[287:306, 986:1001].copy()
        self.assertTrue(has_pink_diamond_outline(region))
        region[:9, :7] = (221, 194, 173)
        self.assertFalse(has_pink_diamond_outline(region))

    def test_gray_looking_transparent_shaman_star_regression(self):
        image = frame('detail-shaman-empty.png')
        region = image[287:306, 986:1001].astype(float)
        self.assertTrue(has_pink_diamond_outline(region))
        self.assertFalse(has_solid_gray_fill(region))
        self.assertEqual(classify_stars(image).slots,
                         ('empty', 'empty', 'empty', 'locked', 'locked'))

    def test_unlocked_pink_border_is_still_empty_not_locked(self):
        image = frame('detail-unlocked-empty.png')
        self.assertTrue(has_solid_gray_fill(image[287:306, 986:1001].astype(float)))
        self.assertEqual(classify_stars(image).slots, ('empty',)*5)

    def test_shaman_candidate_found_after_completed_type(self):
        candidates = visible_candidates(frame('grid-shaman.png', 50, 155))
        self.assertIn((620, 592), [(x, y) for x, y, _ in candidates])
        self.assertTrue(all(stars.level == 0 and stars.locked == 2 for _, _, stars in candidates))

    def test_real_full(self):
        stars = classify_stars(frame('detail-full.png'))
        self.assertTrue(stars.full)
        self.assertEqual(stars.level, 5)
        self.assertEqual(stars.locked, 0)

    def test_real_empty_with_translucent_locked_star(self):
        stars = classify_stars(frame('detail-empty.png'))
        self.assertFalse(stars.full)
        self.assertEqual(stars.level, 0)
        self.assertEqual(stars.locked, 2)

    def test_real_grid_skips_first_full_row(self):
        candidates = visible_candidates(frame('grid.png', 50, 155))
        self.assertEqual(len(candidates), 10)
        self.assertTrue(all(y > 350 and s.level == 0 for _, y, s in candidates))

    def test_unknown_fails_closed(self):
        with self.assertRaises(Stop):
            classify_stars(np.zeros((BASE[1], BASE[0], 3), dtype=np.uint8))

    def test_white_screen_has_no_candidates(self):
        self.assertEqual(visible_candidates(np.full((797, 1416, 3), 255, dtype=np.uint8)), [])

    def test_non_contiguous_stars_fail(self):
        image = frame('detail-empty.png')
        x, y = STAR_CENTERS[1]
        image[y-2:y+3, x-1:x+2] = (40, 200, 250)
        with self.assertRaises(Stop):
            classify_stars(image)

    def test_scrollbar_bottom(self):
        image = np.full((797, 1416, 3), 255, dtype=np.uint8)
        image[200:230, 862:869] = (60, 170, 245)
        self.assertFalse(scrollbar_bottom(image))
        image[640:661, 862:869] = (60, 170, 245)
        self.assertTrue(scrollbar_bottom(image))

    def search_bot(self, image):
        bot = Assistant.__new__(Assistant)
        bot.args = SimpleNamespace(max_scrolls=8)
        bot.win = MagicMock()
        bot.snapshot = MagicMock(return_value=(image, []))
        bot.require = MagicMock()
        return bot

    def test_search_scrolls_before_confirming_bottom(self):
        image = np.full((797, 1416, 3), 255, dtype=np.uint8)
        image[640:661, 862:869] = (60, 170, 245)
        bot = self.search_bot(image)
        self.assertFalse(bot.search())
        self.assertEqual(bot.win.scroll_down.call_count, 2)

    def test_stalled_scroll_is_not_reported_as_not_found(self):
        bot = self.search_bot(np.full((797, 1416, 3), 255, dtype=np.uint8))
        with self.assertRaisesRegex(Stop, '未确认到底'):
            bot.search()
        self.assertEqual(bot.win.scroll_down.call_count, 4)

    def test_ocr_scoped_match(self):
        words = [([[10, 10], [100, 10], [100, 30], [10, 30]], '特别装备强化', .99)]
        self.assertTrue(Assistant.has(words, '特别装备强化', (0, 0, 200, 100)))
        self.assertFalse(Assistant.has(words, '特别装备强化', (500, 500, 1000, 700)))


class LoggingTests(unittest.TestCase):
    def test_stdout_stderr_are_mirrored_and_flushed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.log'
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                with mirror_log(path):
                    print('跳过：材料不足')
                    print('错误：识别失败', file=sys.stderr)
                    text = path.read_text(encoding='utf-8')
                    self.assertIn('跳过：材料不足', text)
                    self.assertIn('错误：识别失败', text)
            self.assertIn('跳过：材料不足', out.getvalue())
            self.assertIn('错误：识别失败', err.getvalue())

    def check_logged_failure(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv', ['upgrade_equipment.py', '--output', directory]), \
                 patch('upgrade_equipment.Assistant', side_effect=failure), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 1)
            files = list(Path(directory).glob('*/run.log'))
            self.assertEqual(len(files), 1)
            text = files[0].read_text(encoding='utf-8')
            self.assertIn('退出码=1', text)
            return text

    def test_guard_stop_is_logged_even_during_initialization(self):
        self.assertIn('已停止：窗口异常', self.check_logged_failure(Stop('窗口异常')))

    def test_unexpected_exception_traceback_is_logged(self):
        text = self.check_logged_failure(RuntimeError('测试异常'))
        self.assertIn('Traceback', text)
        self.assertIn('RuntimeError: 测试异常', text)


if __name__ == '__main__':
    unittest.main()
