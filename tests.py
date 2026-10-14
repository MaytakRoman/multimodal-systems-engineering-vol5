"""Функциональные контракты и независимые контрольные примеры."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mmbook.common import binary_metrics, select_threshold, write_pgm, read_pgm, rng
from mmbook.fusion import late_fusion, contrastive_loss, cross_attention
from mmbook.vision import patchify, restore_box, mask_metrics
from mmbook.vlm import gate_claim, risk_coverage, token_budget
from mmbook.documents import group_rows, validate_invoice, drawing_status
from mmbook.speech import wer, frame_der, energy_vad, TranscriptGate
from mmbook.video import windows, count_crossings, event_intervals
from mmbook.generation import diffuse, flow_euler, distill_linear
from mmbook.media import fit_dynamics, rollout
from mmbook.robotics import shortest_path, simulate
from mmbook.deployment import quantize_rows, low_rank, simulate_queue
from mmbook.industrial import features, train_logistic, predict


class Contracts(unittest.TestCase):
    def test_missing_modality_has_no_influence(self):
        actual=late_fusion([[.8,999],[.2,.8]],[[1,0],[1,1]],[.25,.75])
        np.testing.assert_allclose(actual,[.8,.65])
        with self.assertRaises(ValueError): late_fusion([[.5,.2]],[[0,0]],[.5,.5])

    def test_attention_convex_combination(self):
        output,w=cross_attention([[0,0]],[[1,0],[0,1]],[[2,4],[6,8]])
        np.testing.assert_allclose(output,[[4,6]]); np.testing.assert_allclose(w.sum(axis=1),1)

    def test_contrastive_pairs_and_duplicates(self):
        x=np.eye(4)
        self.assertLess(contrastive_loss(x,x),contrastive_loss(x,np.roll(x,1,axis=0)))
        self.assertAlmostEqual(contrastive_loss(x,x,positive_mask=np.ones((4,4),bool)),0)
        with self.assertRaises(ValueError): contrastive_loss(x,x,0)

    def test_patch_order(self):
        p=patchify(np.arange(16).reshape(4,4),2)
        np.testing.assert_equal(p,[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]])

    def test_coordinate_contract(self):
        np.testing.assert_allclose(restore_box([2,4,6,8],[10,20],2),[11,22,13,24])

    def test_empty_masks_and_overlap(self):
        self.assertEqual(mask_metrics([0,0],[0,0])['iou'],1)
        self.assertAlmostEqual(mask_metrics([1,1,0],[0,1,1])['iou'],1/3)

    def test_budget_and_claim_identity(self):
        self.assertFalse(token_budget(100,40,2,30,10)['fits'])
        status=gate_claim(dict(object_id='b',attribute='red'),dict(id='a',visible=True),dict(value='red',confidence=1))
        self.assertEqual(status,'wrong_object')

    def test_no_accepted_claim_has_undefined_risk(self):
        self.assertIsNone(risk_coverage([True],[.1],[.9])[0]['risk'])

    def test_decimal_invoice(self):
        rows=[dict(qty='3',price='0,10')]
        self.assertEqual(validate_invoice(rows,'0,30')['delta'],'0.00')
        self.assertFalse(validate_invoice(rows,'0,33')['accepted'])

    def test_rows_and_units(self):
        rows=group_rows([dict(x=30,y=22),dict(x=10,y=20),dict(x=10,y=40)],5)
        self.assertEqual([[t['x'] for t in row] for row in rows],[[10,30],[10]])
        self.assertTrue(drawing_status(20,2,-.1,.1,'cm')['accepted'])
        self.assertFalse(drawing_status(20,2,-.1,.1,'mm')['accepted'])

    def test_wer_insertion_deletion(self):
        self.assertAlmostEqual(wer('a b c','a d c x'),2/3)
        self.assertIsNone(wer('','word'))

    def test_speaker_permutation_not_error(self):
        self.assertEqual(frame_der(['a','a','b','b'],['y','y','x','x'])['der'],0)
        d=frame_der(['a',None,'b'],[None,'x','y'])
        self.assertEqual(d['miss'],1); self.assertEqual(d['false_alarm'],1)

    def test_final_revision_exactly_once(self):
        g=TranscriptGate()
        self.assertEqual(g.apply(dict(turn=1,revision=2,final=False,text='new'))['status'],'provisional')
        self.assertEqual(g.apply(dict(turn=1,revision=1,final=True,text='old'))['status'],'stale')
        self.assertEqual(g.apply(dict(turn=1,revision=3,final=True,text='new'))['status'],'committed')
        self.assertEqual(g.apply(dict(turn=1,revision=4,final=True,text='new'))['status'],'already_committed')

    def test_vad_silence(self):
        self.assertEqual(int(energy_vad(np.zeros(640)).sum()),0)

    def test_final_may_confirm_same_revision(self):
        g=TranscriptGate()
        g.apply(dict(turn=2,revision=0,final=False,text='confirmed'))
        self.assertEqual(g.apply(dict(turn=2,revision=0,final=True,text='confirmed'))['status'],'committed')
        other=TranscriptGate(); other.apply(dict(turn=2,revision=0,final=False,text='old'))
        self.assertEqual(other.apply(dict(turn=2,revision=0,final=True,text='new'))['status'],'revision_conflict')

    def test_window_coverage(self):
        w=windows(105,30,10)
        self.assertEqual(w[0][0],0); self.assertEqual(w[-1][1],105)
        covered=set()
        for a,b in w: covered.update(range(a,b))
        self.assertEqual(covered,set(range(105)))

    def test_events_not_frame_count(self):
        self.assertEqual(event_intervals([1,1,1,0,1,0,1,1,1],3),[(0,3),(6,9)])
        self.assertEqual(len(count_crossings({'a':[(0,.1),(1,.9),(2,.1),(3,.9)]})),1)

    def test_diffusion_endpoints(self):
        np.testing.assert_equal(diffuse(np.array([3]),1,np.array([7])),[3])
        np.testing.assert_equal(diffuse(np.array([3]),0,np.array([7])),[7])

    def test_flow_convergence_and_student(self):
        z=rng(11).normal(size=(40,2)); target=np.array([2.,-1.])
        exact=target+(z-target)*np.exp(-2)
        self.assertLess(np.mean((flow_euler(z,target,64)-exact)**2),np.mean((flow_euler(z,target,4)-exact)**2))
        w=distill_linear(z[:20],flow_euler(z[:20],target,64))
        np.testing.assert_allclose(np.c_[z[20:],np.ones(20)]@w,flow_euler(z[20:],target,64),atol=1e-10)

    def test_world_dynamics(self):
        t=np.arange(20); states=np.c_[t,2*t+1]
        w=fit_dynamics(states[:10]); p=rollout(states[9],w,10)
        np.testing.assert_allclose(p,states[9:],atol=1e-8)

    def test_navigation_avoids_wall(self):
        grid=np.zeros((4,4),int); grid[1,0:3]=1
        path=shortest_path(grid,(0,0),(3,0))
        self.assertIsNotNone(path); self.assertTrue(all(grid[p]==0 for p in path))
        self.assertTrue(simulate(grid,(0,0),(3,0),rng(1))['success'])
        grid[1,:]=1; self.assertIsNone(shortest_path(grid,(0,0),(3,0)))

    def test_zero_channel_quantization(self):
        q,s=quantize_rows([[0,0],[1,-1]])
        np.testing.assert_equal(q[0],[0,0]); np.testing.assert_allclose(q*s,[[0,0],[1,-1]],atol=1e-7)
        w=np.array([[1.,2],[2,4]]); np.testing.assert_allclose(low_rank(w,1),w,atol=1e-10)

    def test_queue_drops_new_and_bounds_latency(self):
        q=simulate_queue([0,1,2,10],10,1)
        self.assertEqual(q['dropped'],[1,2]); self.assertEqual([r['completion'] for r in q['accepted']],[10,20])

    def test_metrics_and_validation_threshold(self):
        m=binary_metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,1,1,1))
        self.assertAlmostEqual(m['f1'],.5)
        self.assertAlmostEqual(select_threshold([0,1],[.2,.7]),.7)

    def test_pgm_roundtrip_and_feature_constant(self):
        a=np.arange(16,dtype=np.uint8).reshape(4,4)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.pgm'; write_pgm(p,a); np.testing.assert_equal(read_pgm(p),a)
        np.testing.assert_allclose(features(np.ones((3,3))*100),[0,0,0],atol=1e-12)

    def test_training_restores_from_json_compatible_model(self):
        x=np.array([[-2],[-1],[1],[2]]); y=np.array([0,0,1,1])
        model=train_logistic(x,y)
        np.testing.assert_equal(predict(model,x)>=.5,y)
        model2={k:v.tolist() for k,v in model.items()}
        np.testing.assert_allclose(predict(model2,x),predict(model,x))


if __name__=='__main__': unittest.main(verbosity=2)
