import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_representation_history import family_bootstrap

class FamilyBootstrapTests(unittest.TestCase):
    def test_identical_predictors_have_exact_zero_intervals(self):
        y=np.array([0,1,2,3,1,0]);p=np.full((6,10),.1);games=np.array([1,1,2,3,3,3])
        result=family_bootstrap(y,{'a':p.copy(),'b':p.copy()},p,games,100)
        for scores in result['metrics'].values():
            for entry in scores.values():
                self.assertEqual(entry['variant_minus_reference'],0.)
                self.assertEqual(entry['simultaneous95'],[0.,0.])
                self.assertEqual(entry['pointwise95'],[0.,0.])

    def test_constant_loss_difference_and_contrast_sign(self):
        y=np.zeros(7,dtype=int);games=np.array([1,1,1,2,2,3,4])
        ref=np.full((7,10),.5/9);ref[:,0]=.5
        worse=np.full((7,10),.75/9);worse[:,0]=.25
        result=family_bootstrap(y,{'worse':worse},ref,games,100)
        entry=result['metrics']['log_loss']['worse']
        np.testing.assert_allclose(entry['variant_minus_reference'],np.log(2),atol=1e-14)
        np.testing.assert_allclose(entry['simultaneous95'],[np.log(2)]*2,atol=1e-14)

    def test_duplicate_family_members_do_not_inflate_threshold(self):
        y=np.arange(30)%10;games=np.repeat(np.arange(10),3)
        ref=np.full((30,10),.1)
        rng=np.random.default_rng(91);p=rng.dirichlet(np.ones(10),size=30)
        one=family_bootstrap(y,{'a':p},ref,games,500)
        two=family_bootstrap(y,{'a':p,'b':p},ref,games,500)
        for metric in one['metrics']:
            for field in one['metrics'][metric]['a']:
                np.testing.assert_allclose(one['metrics'][metric]['a'][field],two['metrics'][metric]['a'][field],atol=1e-14,rtol=0)
            self.assertEqual(two['metrics'][metric]['a'],two['metrics'][metric]['b'])

if __name__=='__main__':unittest.main()
