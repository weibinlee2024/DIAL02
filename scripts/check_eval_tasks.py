import sys, re
sys.path.insert(0, '/home/weibin/DIAL-master')

import robocasa
import robosuite
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: triggers registration
from robosuite.environments.base import REGISTERED_ENVS
import gymnasium as gym

# eval.sh 中的任务清单
task_ids = {
    "id (24)": [
        "PnPCupToDrawerClose", "PnPPotatoToMicrowaveClose", "PnPMilkToMicrowaveClose",
        "PnPBottleToCabinetClose", "PnPWineToCabinetClose", "PnPCanToDrawerClose",
        "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
        "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
        "PosttrainPnPNovelFromCuttingboardToPanSplitA",
        "PosttrainPnPNovelFromCuttingboardToPotSplitA",
        "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
        "PosttrainPnPNovelFromPlacematToBasketSplitA",
        "PosttrainPnPNovelFromPlacematToBowlSplitA",
        "PosttrainPnPNovelFromPlacematToPlateSplitA",
        "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
        "PosttrainPnPNovelFromPlateToBowlSplitA",
        "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
        "PosttrainPnPNovelFromPlateToPanSplitA",
        "PosttrainPnPNovelFromPlateToPlateSplitA",
        "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
        "PosttrainPnPNovelFromTrayToPlateSplitA",
        "PosttrainPnPNovelFromTrayToPotSplitA",
        "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
        "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
    ],
    "ood_object_appearance (18)": [
        "EvalPnPNovelFromCuttingboardToBasketSplitB",
        "EvalPnPNovelFromCuttingboardToCardboardboxSplitB",
        "EvalPnPNovelFromCuttingboardToPanSplitB",
        "EvalPnPNovelFromCuttingboardToPotSplitB",
        "EvalPnPNovelFromCuttingboardToTieredbasketSplitB",
        "EvalPnPNovelFromPlacematToBasketSplitB",
        "EvalPnPNovelFromPlacematToBowlSplitB",
        "EvalPnPNovelFromPlacematToPlateSplitB",
        "EvalPnPNovelFromPlacematToTieredshelfSplitB",
        "EvalPnPNovelFromPlateToBowlSplitB",
        "EvalPnPNovelFromPlateToCardboardboxSplitB",
        "EvalPnPNovelFromPlateToPanSplitB",
        "EvalPnPNovelFromPlateToPlateSplitB",
        "EvalPnPNovelFromTrayToCardboardboxSplitB",
        "EvalPnPNovelFromTrayToPlateSplitB",
        "EvalPnPNovelFromTrayToPotSplitB",
        "EvalPnPNovelFromTrayToTieredbasketSplitB",
        "EvalPnPNovelFromTrayToTieredshelfSplitB",
    ],
    "ood_container_combination (14)": [
        "PretrainPnPNovelFromCuttingboardToBowlSplitA",
        "PretrainPnPNovelFromCuttingboardToPlateSplitA",
        "PretrainPnPNovelFromCuttingboardToTieredshelfSplitA",
        "PretrainPnPNovelFromTrayToBasketSplitA",
        "PretrainPnPNovelFromTrayToPanSplitA",
        "PretrainPnPNovelFromTrayToBowlSplitA",
        "PretrainPnPNovelFromPlateToBasketSplitA",
        "PretrainPnPNovelFromPlateToPotSplitA",
        "PretrainPnPNovelFromPlateToTieredshelfSplitA",
        "PretrainPnPNovelFromPlateToTieredbasketSplitA",
        "PretrainPnPNovelFromPlacematToPanSplitA",
        "PretrainPnPNovelFromPlacematToPotSplitA",
        "PretrainPnPNovelFromPlacematToTieredbasketSplitA",
        "PretrainPnPNovelFromPlacematToCardboardboxSplitA",
    ],
    "ood_object_type (32)": [
        "PretrainPnPBaseFromCuttingboardToBowlSplitA",
        "PretrainPnPBaseFromCuttingboardToPlateSplitA",
        "PretrainPnPBaseFromCuttingboardToTieredshelfSplitA",
        "PretrainPnPBaseFromTrayToBasketSplitA",
        "PretrainPnPBaseFromTrayToPanSplitA",
        "PretrainPnPBaseFromTrayToBowlSplitA",
        "PretrainPnPBaseFromPlateToBasketSplitA",
        "PretrainPnPBaseFromPlateToPotSplitA",
        "PretrainPnPBaseFromPlateToTieredshelfSplitA",
        "PretrainPnPBaseFromPlateToTieredbasketSplitA",
        "PretrainPnPBaseFromPlacematToPanSplitA",
        "PretrainPnPBaseFromPlacematToPotSplitA",
        "PretrainPnPBaseFromPlacematToTieredbasketSplitA",
        "PretrainPnPBaseFromPlacematToCardboardboxSplitA",
        "PretrainPnPBaseFromCuttingboardToPotSplitA",
        "PretrainPnPBaseFromCuttingboardToBasketSplitA",
        "PretrainPnPBaseFromCuttingboardToTieredbasketSplitA",
        "PretrainPnPBaseFromCuttingboardToPanSplitA",
        "PretrainPnPBaseFromCuttingboardToCardboardboxSplitA",
        "PretrainPnPBaseFromPlacematToBowlSplitA",
        "PretrainPnPBaseFromPlacematToPlateSplitA",
        "PretrainPnPBaseFromPlacematToBasketSplitA",
        "PretrainPnPBaseFromPlacematToTieredshelfSplitA",
        "PretrainPnPBaseFromPlateToPanSplitA",
        "PretrainPnPBaseFromPlateToCardboardboxSplitA",
        "PretrainPnPBaseFromPlateToBowlSplitA",
        "PretrainPnPBaseFromPlateToPlateSplitA",
        "PretrainPnPBaseFromTrayToTieredshelfSplitA",
        "PretrainPnPBaseFromTrayToPlateSplitA",
        "PretrainPnPBaseFromTrayToTieredbasketSplitA",
        "PretrainPnPBaseFromTrayToCardboardboxSplitA",
        "PretrainPnPBaseFromTrayToPotSplitA",
    ],
}

all_registered = set(REGISTERED_ENVS.keys())
print(f"robosuite REGISTERED_ENVS 总数: {len(all_registered)}")
missing_all = {}
for cat, names in task_ids.items():
    missing = []
    for n in names:
        if n not in all_registered:
            missing.append(n)
    missing_all[cat] = missing
    print(f"{cat}: 共 {len(names)} 个, 缺失 {len(missing)} 个 {missing if missing else ''}")

total_missing = sum(len(v) for v in missing_all.values())
print(f"\n总缺失: {total_missing} 个任务")
