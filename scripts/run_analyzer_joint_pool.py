#!/usr/bin/env python3
"""Run the frozen V5 analyzer with preregistered public task bindings."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from wam_reranking import TaskBinding

source = Path(os.environ["FROZEN_ANALYZER"]).resolve()
spec = importlib.util.spec_from_file_location("frozen_v5_analyzer", source)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.TASK_BINDINGS.update({
    11: TaskBinding(11, "open the top drawer of the cabinet", "wooden_cabinet_1", "cabinet"),
    18: TaskBinding(18, "put the frying pan on the stove", "chefmate_8_frypan_1", "flat_stove_1"),
    27: TaskBinding(27, "put the wine bottle on the wine rack", "wine_bottle_1", "wine_rack_1"),
    35: TaskBinding(35, "open the microwave", "microwave_1", "microwave_frame"),
    46: TaskBinding(46, "pick up the alphabet soup and put it in the basket", "alphabet_soup_1", "basket_1"),
    57: TaskBinding(57, "pick up the cream cheese and put it in the tray", "cream_cheese_1", "wooden_tray_1"),
    68: TaskBinding(68, "put the yellow and white mug on the right plate", "white_yellow_mug_1", "plate_2"),
    86: TaskBinding(86, "pick up the book in the middle and place it on the cabinet shelf", "black_book_1", "wooden_two_layer_shelf_1"),
    13: TaskBinding(13, "put the black bowl at the front on the plate", "akita_black_bowl_1", "plate_1"),
    32: TaskBinding(32, "put the ketchup in the top drawer of the cabinet", "ketchup_1", "wooden_cabinet_1_top_level"),
    34: TaskBinding(34, "put the yellow and white mug to the front of the white mug", "white_yellow_mug_1", "white_coffee_mug_1"),
    16: TaskBinding(16, "stack the black bowl at the front on the black bowl in the middle", "akita_black_bowl_1", "akita_black_bowl_2"),
    37: TaskBinding(37, "put the white bowl to the right of the plate", "white_bowl_1", "plate_1"),
    63: TaskBinding(63, "stack the left bowl on the right bowl and place them in the tray", "akita_black_bowl_1", "akita_black_bowl_2"),
    73: TaskBinding(73, "pick up the book and place it in the front compartment of the caddy", "black_book_1", "caddy_1"),
})
module.main()
