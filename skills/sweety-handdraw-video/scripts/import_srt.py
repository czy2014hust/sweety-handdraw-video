#!/usr/bin/env python3
"""Write narration alignment from an SRT whose text already matches the script.

Timestamps come from the subtitle cues. A cue that crosses a beat boundary is
rejected instead of being split by character count. Scenes are not regrouped
into fixed 25–35 second blocks.
"""
import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

from narration_master import script_text, sha256

_TIME = re.compile(r'(\d+):(\d{2}):(\d{2})[,.](\d{1,3})')


def _to_seconds(hours, minutes, seconds, millis):
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) + int(millis.ljust(3, '0')) / 1000


def parse_srt(text):
    """Parse SRT cues. Blank lines and a leading BOM are ignored. Line breaks inside a cue are removed."""
    text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    cues = []
    for block in re.split(r'\n\s*\n', text.strip()):
        lines = [line for line in block.split('\n') if line.strip()]
        time_index = next((index for index, line in enumerate(lines) if '-->' in line), None)
        if time_index is None:
            continue
        times = _TIME.findall(lines[time_index])
        if len(times) < 2:
            continue
        start, end = _to_seconds(*times[0]), _to_seconds(*times[1])
        body = ''.join(line.strip() for line in lines[time_index + 1:])
        if not body or end <= start:
            raise ValueError('Each SRT cue needs text and a positive duration')
        if cues and start < cues[-1]['end']:
            raise ValueError('SRT cues overlap; fix the timeline before importing')
        cues.append({'text': body, 'start': start, 'end': end})
    if not cues:
        raise ValueError('No SRT cues found')
    return cues


def align_cues(cues, beats):
    """Map whole cues onto beats. Cue text must concatenate to each beat exactly."""
    pending = list(cues)
    aligned = []
    for beat in beats:
        used = []
        text = ''
        while text != beat['text']:
            if not pending or not beat['text'].startswith(text + pending[0]['text']):
                preview = pending[0]['text'] if pending else ''
                raise ValueError(f'SRT text does not land on beat {beat["id"]}: have {text!r}, next cue {preview!r}, beat {beat["text"]!r}')
            used.append(pending.pop(0))
            text += used[-1]['text']
        start = used[0]['start']
        end = used[-1]['end']
        aligned.append({'id': beat['id'], 'text': beat['text'], 'start': start, 'end': end,
                        'caption_cues': [{'text': cue['text'], 'start': round(cue['start'] - start, 3),
                                          'end': round(cue['end'] - start, 3)} for cue in used]})
    if pending:
        raise ValueError('SRT has text past the last beat: ' + pending[0]['text'])
    for previous, current in zip(aligned, aligned[1:]):
        if current['start'] < previous['end']:
            raise ValueError('Beat ranges overlap after SRT import')
    return aligned


def wav_duration(path):
    with wave.open(str(path)) as stream:
        return stream.getnframes() / stream.getframerate()


def import_srt(job, srt_path):
    root = Path(job).resolve()
    manifest = json.loads((root / 'job.json').read_text())
    settings = manifest.get('narration') or {}
    if not settings.get('source') or not settings.get('alignment'):
        raise ValueError('job.json needs narration.source and narration.alignment')
    source = (root / settings['source']).resolve()
    destination = (root / settings['alignment']).resolve()
    if not source.is_file():
        raise ValueError('Narration master is missing: ' + settings['source'])
    if not destination.is_relative_to(root) or not source.is_relative_to(root):
        raise ValueError('Narration files must stay inside the job')
    beats = [beat for page in manifest['pages'] for beat in page['beats']]
    cues = parse_srt(Path(srt_path).read_text(encoding='utf-8-sig'))
    aligned = align_cues(cues, beats)
    duration = wav_duration(source)
    if aligned[-1]['end'] > duration + 1 / 48000:
        raise ValueError('SRT ends after the narration master')
    text = script_text(manifest)
    alignment = {'version': 1, 'source_sha256': sha256(source),
                 'script_sha256': hashlib.sha256(text.encode()).hexdigest(),
                 'method': 'srt_cues', 'beats': aligned}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(alignment, ensure_ascii=False, indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--srt', required=True, type=Path)
    arguments = parser.parse_args()
    print(import_srt(arguments.job, arguments.srt))


if __name__ == '__main__':
    main()
