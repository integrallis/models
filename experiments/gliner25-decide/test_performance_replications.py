import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from run_performance_pairs import summarize_pairs


class ReplicationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cases=[dict(id=str(i),bucket='shape',phase='warmup' if i<3 else 'timed') for i in range(33)]

    def records(self,count=3,slow_pair=None,changed_pair=None):
        records=[]
        for index in range(count):
            record=dict(order=['reference','models'] if index%2==0 else ['models','reference'])
            for arm in record['order']:
                path=self.root/f'{index}-{arm}.jsonl'
                ready=dict(type='ready',device=['cpu'],snapshot='pinned',threads=8,
                    backend=arm,dtype=['float32'],driverHostname='host',driverCpuAffinity=list(range(8)))
                if index==changed_pair and arm=='models':ready['nativeSha256']='changed'
                ms=2 if index==slow_pair and arm=='models' else 1
                rows=[ready]+[dict(id=c['id'],wallMs=ms,elapsedMs=ms,peakRssKiB=1024,selected={'task':['label']}) for c in self.cases]
                data=''.join(json.dumps(r)+'\n' for r in rows).encode();path.write_bytes(data)
                record[arm]=dict(file=path.name,sha256=hashlib.sha256(data).hexdigest())
            records.append(record)
        return records

    def test_requires_three_complete_pairs_and_never_grants_catalog(self):
        screening=summarize_pairs(self.cases,self.root,self.records(1))
        self.assertTrue(screening['passed']);self.assertFalse(screening['replicationGatePassed'])
        final=summarize_pairs(self.cases,self.root,self.records())
        self.assertTrue(final['replicationGatePassed']);self.assertFalse(final['catalogQualified'])
        self.assertEqual(0,final['buckets']['shape']['throughputRatio']['sampleStandardDeviation'])

    def test_one_failing_pair_is_not_averaged_away(self):
        report=summarize_pairs(self.cases,self.root,self.records(slow_pair=1))
        self.assertFalse(report['replicationGatePassed'])

    def test_rejects_changed_native_identity_order_or_raw_bytes(self):
        with self.assertRaises(ValueError):summarize_pairs(self.cases,self.root,self.records(changed_pair=1))
        records=self.records();records[1]['order'].reverse()
        with self.assertRaises(ValueError):summarize_pairs(self.cases,self.root,records)
        records=self.records();(self.root/records[0]['models']['file']).write_text('changed')
        with self.assertRaises(ValueError):summarize_pairs(self.cases,self.root,records)


if __name__=='__main__':unittest.main()
