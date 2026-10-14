import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('finishcore',ROOT/'scripts/finish_core_scene.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def complete_inputs():
    status={'complete':True,'trained_exported_core_count':28,'planned_core_ids':list(range(28)),
            'cores':{'core_%03d'%i:{'status':'trained_exported'} for i in range(28)}}
    plan={'sources':[{'core_ids':['block_%03d'%i]} for i in range(28)]}
    return status,plan


def test_requires_actual_all28_not_successful_subset():
    status,plan=complete_inputs();module.require_full_batch(status,plan)
    status['cores']['core_027']['status']='prepared'
    with pytest.raises(ValueError,match='All28'):module.require_full_batch(status,plan)


def test_rejects_duplicated_core_and_missing_other_core():
    status,plan=complete_inputs();plan['sources'][-1]['core_ids']=['block_026']
    with pytest.raises(ValueError,match='exactly once'):module.require_full_batch(status,plan)


def test_reads_identity_of_actual_current_process():
    import os
    value=module.identity(os.getpid())
    assert value['pid']==os.getpid() and value['start_ticks'] and value['argv']
