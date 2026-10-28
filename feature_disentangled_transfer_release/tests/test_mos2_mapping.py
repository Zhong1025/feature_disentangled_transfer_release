import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_transfer.mos2 import map_mos_to_cnt15, substrate_lattice_constant, substrate_thermal_conductivity


class SubstrateMappingTests(unittest.TestCase):
    def test_oxide_names_do_not_match_elemental_silicon(self):
        for name in ("SiO2", "SiO2/Si", "SiO2/Si with glass lamella nearby", "silica", "silicon dioxide", "quartz"):
            with self.subTest(name=name):
                self.assertEqual(substrate_thermal_conductivity(name), 1.4)
                self.assertEqual(substrate_lattice_constant(name), 4.91)

    def test_silicon_and_sic_keep_their_own_values(self):
        self.assertEqual(substrate_thermal_conductivity("silicon wafer"), 149.0)
        self.assertEqual(substrate_thermal_conductivity("Si wafer"), 149.0)
        self.assertEqual(substrate_thermal_conductivity("SiC"), 120.0)
        self.assertEqual(substrate_thermal_conductivity("Dragontrail alkali-aluminosilicate glass"), 1.0)
        self.assertEqual(substrate_thermal_conductivity("molten soda-lime-silica glass on Mo foil"), 1.0)
        self.assertEqual(substrate_thermal_conductivity("silica glass"), 1.4)
        self.assertTrue(np.isnan(substrate_lattice_constant("silica glass")))
        self.assertTrue(np.isnan(substrate_lattice_constant("Dragontrail alkali-aluminosilicate glass")))
        self.assertTrue(np.isnan(substrate_lattice_constant("molten soda-lime-silica glass on Mo foil")))
        self.assertTrue(np.isnan(substrate_thermal_conductivity("molten glass; c-plane sapphire")))
        self.assertEqual(substrate_lattice_constant("molten glass; c-plane sapphire"), 4.758)
        self.assertTrue(np.isnan(substrate_thermal_conductivity("Si3N4")))
        self.assertTrue(np.isnan(substrate_thermal_conductivity("silicon nitride")))
        self.assertEqual(substrate_lattice_constant("silicon wafer"), 5.431)
        self.assertEqual(substrate_lattice_constant("Si wafer"), 5.431)
        self.assertTrue(np.isnan(substrate_lattice_constant("Si3N4")))
        self.assertTrue(np.isnan(substrate_lattice_constant("silicon nitride")))

    def test_mapper_applies_corrected_oxide_value(self):
        substrates = ["SiO2", "silica", "silica glass", "SiO2/Si", "silicon dioxide", "SiO2/Si with glass lamella nearby"]
        n = len(substrates)
        raw = pd.DataFrame({
            "growth_temperature_C": [700.0] * n,
            "total_flow_sccm": [100.0] * n,
            "ar_flow_sccm": [100.0] * n,
            "n2_flow_sccm": [0.0] * n,
            "h2_flow_sccm": [0.0] * n,
            "h2s_flow_sccm": [np.nan] * n,
            "o2_flow_sccm": [0.0] * n,
            "s_source_temperature_C": [170.0] * n,
            "precursor_ratio": [1.0] * n,
            "mo_precursor": ["MoO3"] * n,
            "promoter_type": ["none"] * n,
            "promoter_used": [0] * n,
            "substrate": substrates,
            "substrate_temperature_C": [25.0] * len(substrates),
            "holding_time_min": [60.0] * len(substrates),
        })
        mapped = map_mos_to_cnt15(raw)
        np.testing.assert_array_equal(mapped["Thermal Conductivity"].to_numpy(), [1.4] * len(substrates))
        np.testing.assert_array_equal(mapped["Support_latticeConstant"].to_numpy(), [4.91, 4.91, np.nan, 4.91, 4.91, 4.91])


if __name__ == "__main__":
    unittest.main()
