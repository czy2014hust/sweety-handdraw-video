"""SRT import keeps cue timestamps and refuses to invent time inside a cue."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from import_srt import align_cues, import_srt, parse_srt
from narration_master import load_master
from video_job import save_wave


class ImportSrtTests(unittest.TestCase):
    def test_multiline_cues_keep_their_own_timestamps(self):
        cues = parse_srt('1\n00:00:00,000 --> 00:00:01,200\n你好\n\n2\n00:00:01.400 --> 00:00:02,500\n世界。\n')
        aligned = align_cues(cues, [{'id': 'one', 'text': '你好世界。'}])
        self.assertEqual(aligned[0]['start'], 0)
        self.assertEqual(aligned[0]['end'], 2.5)
        self.assertEqual(aligned[0]['caption_cues'][1]['start'], 1.4)
        self.assertEqual(''.join(cue['text'] for cue in aligned[0]['caption_cues']), '你好世界。')

    def test_a_cue_crossing_a_beat_is_not_split_by_character_count(self):
        cues = parse_srt('1\n00:00:00,000 --> 00:00:04,000\n你好世界。\n')
        with self.assertRaisesRegex(ValueError, 'does not land on beat'):
            align_cues(cues, [{'id': 'a', 'text': '你好'}, {'id': 'b', 'text': '世界。'}])

    def test_import_writes_alignment_the_master_loader_accepts(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        tone = (np.sin(np.arange(3 * 48000) / 48000 * 2 * np.pi * 220) * 7000).astype(np.int16)
        save_wave(root / 'master.wav', tone)
        (root / 'captions.srt').write_text('1\n00:00:00,100 --> 00:00:01,000\n前一句。\n\n2\n00:00:01,200 --> 00:00:02,400\n后一句。\n', encoding='utf-8')
        beats = [{'id': 'a', 'text': '前一句。'}, {'id': 'b', 'text': '后一句。'}]
        manifest = {'pages': [{'beats': [dict(beat, actions=[]) for beat in beats]}],
                    'narration': {'source': 'master.wav', 'alignment': 'alignment.json'},
                    'audio': {'end_hold_seconds': 0.45}}
        (root / 'job.json').write_text(json.dumps(manifest))
        self.assertEqual(import_srt(root, root / 'captions.srt'), root / 'alignment.json')
        _, loaded, _ = load_master(root, manifest, root / 'converted.wav')
        self.assertEqual(loaded['b']['start'], 1.2)
        temporary.cleanup()

    def test_overlapping_cues_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            parse_srt('1\n00:00:00,000 --> 00:00:01,000\n甲\n\n2\n00:00:00,500 --> 00:00:01,500\n乙\n')


if __name__ == '__main__':
    unittest.main()
