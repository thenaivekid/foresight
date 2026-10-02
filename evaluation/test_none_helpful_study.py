"""CPU-only safeguards for none/helpful selection, manifests and full-run gating."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ALL_TASKS
from none_helpful_study import choose_samples, full_guard, write_subset
from repeat_vision_model_only import load_run
from vision_model_only import validate_input_manifest


def population():
    return [{'id':f'{task}::{dep}{i}', 'video_id':f'{task}_{dep}{i}',
             'task':task, 'audio_dependency':dep, 'duration':60+i,
             'video_path':'unused.mp4', 'ground_truth':[{'t_sec':20,'count':1}]}
            for task in ALL_TASKS for dep in ('none','helpful','required') for i in range(4)]


class TestStudySelection(unittest.TestCase):
    def test_balanced_video_disjoint_selection_and_exclusion(self):
        records = population()
        excluded = {r['video_id'] for r in records if r['video_id'].endswith('none0')}
        split = choose_samples(records, excluded)
        self.assertEqual(len(split['calibration']),18)
        self.assertEqual(len(split['validation']),18)
        for phase, rows in split.items():
            self.assertEqual({(r['task'],r['audio_dependency']) for r in rows},
                             {(t,d) for t in ALL_TASKS for d in ('none','helpful')})
        c = {r['video_id'] for r in split['calibration']}
        v = {r['video_id'] for r in split['validation']}
        self.assertFalse(c & v)
        self.assertFalse(v & excluded)

    def test_selection_does_not_use_gt_or_prediction_values(self):
        rows = population()
        first = choose_samples(rows,set())
        changed = copy.deepcopy(rows)
        for r in changed:
            r['ground_truth'] = [{'t_sec':100000,'count':9999}]
            r['predictions'] = [{'raw':'test leak','t_sec':0}]
        second = choose_samples(changed,set())
        for phase in first:
            self.assertEqual([r['id'] for r in first[phase]], [r['id'] for r in second[phase]])

    def test_selection_is_reproducible_and_order_independent(self):
        rows = population()
        a, b = choose_samples(rows,set()), choose_samples(list(reversed(rows)),set())
        self.assertEqual(a,b)

    def test_infeasible_bucket_fails_instead_of_dropping_task(self):
        rows = [r for r in population() if not(r['task']==ALL_TASKS[0] and r['audio_dependency']=='helpful')]
        with self.assertRaises(ValueError):
            choose_samples(rows,set())

    def test_cross_task_video_reuse_is_not_split_leakage(self):
        rows = population()
        for r in rows:
            if r['task'] in ALL_TASKS[:2] and r['audio_dependency']=='none':
                r['video_id'] = 'shared_'+r['id'][-1]
        split = choose_samples(rows,set())
        ids = [r['video_id'] for phase in split.values() for r in phase]
        self.assertEqual(len(ids),len(set(ids)))


class TestManifestContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_shards_cover_full_none_helpful_subset_without_duplicates(self):
        rows = [r for r in population() if r['audio_dependency']!='required']
        write_subset(self.root,rows,Path('/dataset'),'full')
        ids = []
        for shard in range(4):
            manifest, part = validate_input_manifest(self.root,shard)
            self.assertEqual(manifest['audio_filter'],'none_helpful')
            ids += [r['id'] for r in part]
        self.assertEqual(set(ids),{r['id'] for r in rows})
        self.assertEqual(len(ids),len(set(ids)))

    def test_required_sample_is_rejected_before_model_load(self):
        rows = population()[:4]
        write_subset(self.root,rows,Path('/dataset'),'test')
        path = self.root/'shard0.json'
        shard = json.loads(path.read_text())
        shard[0]['audio_dependency'] = 'required'
        path.write_text(json.dumps(shard))
        with self.assertRaises(ValueError):
            validate_input_manifest(self.root,0)

    def test_arbitrary_completed_subset_size_not_hardcoded_18(self):
        rows = [{'id':'a','audio_dependency':'none'}, {'id':'b','audio_dependency':'helpful'}]
        (self.root/'manifest.json').write_text(json.dumps({'ids':['a','b'],'audio_filter':'none_helpful'}))
        dest = self.root/'g0'
        dest.mkdir()
        for row in rows:
            row.update(gate_trace=[{'t_sec':1,'p_hit':.1}],effective_config={'deterministic':True})
        (dest/'online_pred.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        self.assertEqual(len(load_run(self.root)),2)

    def test_empty_manifest_is_not_vacuously_complete(self):
        (self.root/'manifest.json').write_text(json.dumps({'ids':[],'audio_filter':'none_helpful'}))
        with self.assertRaises(ValueError):
            load_run(self.root)

    def test_missing_live_delivery_is_never_backdated(self):
        (self.root/'manifest.json').write_text(json.dumps({'ids':['a'],'audio_filter':'none_helpful'}))
        dest=self.root/'g0'; dest.mkdir()
        row={'id':'a','audio_dependency':'helpful','gate_trace':[{'t_sec':1,'p_hit':.9}],
             'effective_config':{'deterministic':False},'predictions':[{'t_sec':1,'raw':'x'}]}
        (dest/'online_pred.jsonl').write_text(json.dumps(row)+'\n')
        with self.assertRaises(KeyError):
            load_run(self.root)

    def test_full_run_requires_successful_matching_frozen_config(self):
        (self.root/'frozen_config.json').write_text(json.dumps({'arm':'ev0'}))
        with patch('none_helpful_study.check_provenance',return_value={}):
            with self.assertRaises(FileNotFoundError):
                full_guard(self.root)
            for record in ({'passed':False,'arm':'ev0'}, {'passed':True,'arm':'raw'}):
                (self.root/'validation_decision.json').write_text(json.dumps(record))
                with self.assertRaises(RuntimeError):
                    full_guard(self.root)
            (self.root/'validation_decision.json').write_text(json.dumps({'passed':True,'arm':'ev0'}))
            self.assertEqual(full_guard(self.root),'ev0')


if __name__ == '__main__':
    unittest.main(verbosity=2)