"""
Test suite for psgscoring.profiles

Run with: python -m pytest test_profiles.py -v
"""

import warnings
import pytest
from psgscoring.profiles import (
    Profile,
    HypopneaRules,
    ApneaRules,
    SpO2Rules,
    PostProcessingRules,
    get_profile,
    list_profiles,
    list_profile_groups,
    resolve_profile_name,
    profile_metadata,
    PROFILES,
    PROFILE_GROUPS,
    LEGACY_ALIASES,
)


# ============================================================
# Structural tests
# ============================================================

class TestRegistry:
    """Test the profile registry is correctly populated."""

    def test_nine_profiles_exist(self):
        assert len(PROFILES) == 8  # 6 standard + strict/sensitive variants = 8

    def test_all_profiles_valid(self):
        for name, p in PROFILES.items():
            assert p.name == name, f"name mismatch for {name}"
            assert p.display_name, f"missing display_name for {name}"
            assert p.family in ("clinical", "dataset", "legacy", "exploratory")
            assert p.aasm_version, f"missing aasm_version for {name}"
            assert p.aasm_rule, f"missing aasm_rule for {name}"
            assert p.description, f"missing description for {name}"
            assert p.citation, f"missing citation for {name}"

    def test_all_profiles_have_hypopnea_rules(self):
        for name, p in PROFILES.items():
            assert isinstance(p.hypopnea, HypopneaRules)
            assert 0 < p.hypopnea.flow_reduction_threshold <= 1.0
            assert p.hypopnea.min_duration_s >= 10.0
            assert p.hypopnea.max_duration_s >= p.hypopnea.min_duration_s

    def test_apnea_defaults_consistent(self):
        """All profiles should share the same apnea rules (≥90%, thermistor)."""
        for name, p in PROFILES.items():
            assert p.apnea.flow_reduction_threshold == 0.90
            assert p.apnea.sensor == "thermistor"
            assert p.apnea.min_duration_s == 10.0


# ============================================================
# Parameter-specific tests per profile
# ============================================================

class TestAASMv3Rec:
    """AASM v3 Rule 1A — the DEFAULT clinical profile."""

    def test_exists(self):
        p = get_profile("aasm_v3_rec")
        assert p.name == "aasm_v3_rec"

    def test_flow_threshold_30pct(self):
        p = get_profile("aasm_v3_rec")
        assert p.hypopnea.flow_reduction_threshold == 0.30

    def test_desat_threshold_3pct(self):
        p = get_profile("aasm_v3_rec")
        assert p.hypopnea.desat_threshold == 0.03

    def test_desat_or_arousal(self):
        """v3 Rule 1A: desat OR arousal is sufficient."""
        p = get_profile("aasm_v3_rec")
        assert p.hypopnea.desat_or_arousal is True
        assert p.hypopnea.desat_required is False

    def test_nasal_pressure_sensor(self):
        p = get_profile("aasm_v3_rec")
        assert p.hypopnea.sensor == "nasal_pressure"

    def test_aasm_version_labeled_v3(self):
        p = get_profile("aasm_v3_rec")
        assert "v3" in p.aasm_version
        assert "2023" in p.aasm_version


class TestAASMv2Rec:
    """AASM v2 Rule 1A — functionally identical to v3_rec."""

    def test_same_flow_threshold_as_v3(self):
        p2 = get_profile("aasm_v2_rec")
        p3 = get_profile("aasm_v3_rec")
        assert p2.hypopnea.flow_reduction_threshold == p3.hypopnea.flow_reduction_threshold

    def test_same_desat_rule_as_v3(self):
        p2 = get_profile("aasm_v2_rec")
        p3 = get_profile("aasm_v3_rec")
        assert p2.hypopnea.desat_threshold == p3.hypopnea.desat_threshold
        assert p2.hypopnea.desat_or_arousal == p3.hypopnea.desat_or_arousal

    def test_version_label_differs(self):
        p = get_profile("aasm_v2_rec")
        assert "v2" in p.aasm_version


class TestAASMv1Rec:
    """AASM v1 2007 — historical, 4% desat, no arousal alternative."""

    def test_flow_threshold_30pct(self):
        p = get_profile("aasm_v1_rec")
        assert p.hypopnea.flow_reduction_threshold == 0.30

    def test_desat_threshold_4pct_not_3pct(self):
        """Key distinction from v2/v3."""
        p = get_profile("aasm_v1_rec")
        assert p.hypopnea.desat_threshold == 0.04

    def test_desat_required_no_arousal_alternative(self):
        """v1 does NOT allow arousal as qualifier."""
        p = get_profile("aasm_v1_rec")
        assert p.hypopnea.desat_required is True
        assert p.hypopnea.desat_or_arousal is False

    def test_year_2007_labeled(self):
        p = get_profile("aasm_v1_rec")
        assert "2007" in p.aasm_version


class TestCMSMedicare:
    """CMS / Medicare — AASM Rule 1B (now OPTIONAL)."""

    def test_flow_threshold_30pct(self):
        p = get_profile("cms_medicare")
        assert p.hypopnea.flow_reduction_threshold == 0.30

    def test_desat_threshold_4pct(self):
        p = get_profile("cms_medicare")
        assert p.hypopnea.desat_threshold == 0.04

    def test_no_arousal_qualifier(self):
        p = get_profile("cms_medicare")
        assert p.hypopnea.desat_required is True
        assert p.hypopnea.desat_or_arousal is False

    def test_rule_1b_labeled(self):
        p = get_profile("cms_medicare")
        assert "1B" in p.aasm_rule


class TestMESAShhs:
    """NSRR dataset reproduction profile.

    v0.5.0 metadata correction: previously the profile was
    declared as `rip_bands_primary` with `desat_threshold=None`,
    but the MESA Sleep PSG Scoring Manual identifies hypopneas from
    airflow signals (nasal pressure primary, with thermistor and
    RIP-bands as supporting context) and the canonical NSRR clinical
    AHI is `nsrr_ahi_hp3u` (3% desat OR arousal). The profile
    metadata now reflects that.
    """

    def test_nasal_pressure_primary(self):
        """MESA Brigham Reading Center scores hypopneas from nasal pressure
        as primary airflow signal (per the MESA Sleep PSG Scoring Manual)."""
        p = get_profile("mesa_shhs")
        assert p.hypopnea.sensor == "nasal_pressure_primary"

    def test_desat_or_arousal_gating(self):
        """Canonical MESA clinical AHI (`nsrr_ahi_hp3u`) gates on
        3% desat OR arousal, equivalent to AASM v2 Rule 1A."""
        p = get_profile("mesa_shhs")
        assert p.hypopnea.desat_threshold == 0.03
        assert p.hypopnea.desat_required is False
        assert p.hypopnea.desat_or_arousal is True

    def test_emits_multiple_ahi_variants(self):
        p = get_profile("mesa_shhs")
        assert len(p.hypopnea.output_variants) >= 3
        assert "ahi_3pct" in p.hypopnea.output_variants
        assert "ahi_3pct_arousal" in p.hypopnea.output_variants
        assert "ahi_4pct" in p.hypopnea.output_variants

    def test_unsure_as_hypopnea_flag(self):
        """NSRR 'Unsure' tag handling."""
        p = get_profile("mesa_shhs")
        assert p.post_processing.unsure_as_hypopnea is True

    def test_family_is_dataset(self):
        p = get_profile("mesa_shhs")
        assert p.family == "dataset"


class TestChicago1999:
    """Chicago Criteria — pre-AASM, ≥50% threshold."""

    def test_flow_threshold_50pct(self):
        """Defining feature: stricter than AASM."""
        p = get_profile("chicago_1999")
        assert p.hypopnea.flow_reduction_threshold == 0.50

    def test_desat_threshold_3pct(self):
        p = get_profile("chicago_1999")
        assert p.hypopnea.desat_threshold == 0.03

    def test_desat_or_arousal(self):
        p = get_profile("chicago_1999")
        assert p.hypopnea.desat_or_arousal is True

    def test_family_is_legacy(self):
        p = get_profile("chicago_1999")
        assert p.family == "legacy"


class TestExploratoryVariants:
    """Strict and Sensitive variants of v3 RECOMMENDED."""

    def test_strict_lower_stability_cv(self):
        strict = get_profile("aasm_v3_strict")
        rec = get_profile("aasm_v3_rec")
        assert strict.post_processing.stability_filter_cv < rec.post_processing.stability_filter_cv

    def test_strict_no_flow_smoothing(self):
        p = get_profile("aasm_v3_strict")
        assert p.post_processing.flow_smoothing_s == 0

    def test_strict_no_breath_level(self):
        p = get_profile("aasm_v3_strict")
        assert p.post_processing.breath_level_detection is False

    def test_sensitive_lower_flow_threshold(self):
        """Sensitive: 25% instead of 30%."""
        sens = get_profile("aasm_v3_sensitive")
        assert sens.hypopnea.flow_reduction_threshold == 0.25

    def test_sensitive_higher_stability_cv(self):
        sens = get_profile("aasm_v3_sensitive")
        rec = get_profile("aasm_v3_rec")
        assert sens.post_processing.stability_filter_cv > rec.post_processing.stability_filter_cv


# ============================================================
# Comparison tests (cross-profile consistency)
# ============================================================

class TestCrossProfileConsistency:
    """Relationships that should always hold between profiles."""

    def test_v1_stricter_than_v3_in_desat(self):
        """v1 requires 4%, v3 requires 3% or arousal → v1 is stricter."""
        v1 = get_profile("aasm_v1_rec")
        v3 = get_profile("aasm_v3_rec")
        assert v1.hypopnea.desat_threshold > v3.hypopnea.desat_threshold

    def test_cms_same_desat_as_v1(self):
        """Both v1 and CMS use 4% without arousal alternative."""
        v1 = get_profile("aasm_v1_rec")
        cms = get_profile("cms_medicare")
        assert v1.hypopnea.desat_threshold == cms.hypopnea.desat_threshold
        assert v1.hypopnea.desat_or_arousal == cms.hypopnea.desat_or_arousal

    def test_chicago_strictest_flow(self):
        """Chicago 50% vs AASM 30%."""
        chi = get_profile("chicago_1999")
        for name in ["aasm_v1_rec", "aasm_v2_rec", "aasm_v3_rec", "cms_medicare"]:
            p = get_profile(name)
            assert chi.hypopnea.flow_reduction_threshold > p.hypopnea.flow_reduction_threshold

    def test_sensitive_least_strict_flow(self):
        """Sensitive 25% is the most permissive."""
        sens = get_profile("aasm_v3_sensitive")
        assert sens.hypopnea.flow_reduction_threshold == 0.25


# ============================================================
# Legacy alias / deprecation tests
# ============================================================

class TestLegacyAliases:

    def test_standard_maps_to_v3_rec(self):
        assert LEGACY_ALIASES["standard"] == "aasm_v3_rec"

    def test_strict_maps_to_v3_strict(self):
        assert LEGACY_ALIASES["strict"] == "aasm_v3_strict"

    def test_sensitive_maps_to_v3_sensitive(self):
        assert LEGACY_ALIASES["sensitive"] == "aasm_v3_sensitive"

    def test_legacy_name_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = get_profile("standard")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert p.name == "aasm_v3_rec"

    def test_legacy_strict_resolves(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = get_profile("strict")
        assert p.name == "aasm_v3_strict"

    def test_new_name_no_warning(self):
        """Direct use of new name should NOT warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = get_profile("aasm_v3_rec")
            # Filter to DeprecationWarnings only
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0


# ============================================================
# Profile groups
# ============================================================

class TestProfileGroups:

    def test_clinical_group_matches_paper_v31(self):
        """Paper v31 uses exactly these three profiles for confidence interval."""
        clinical = PROFILE_GROUPS["clinical"]
        assert clinical == ["aasm_v3_strict", "aasm_v3_rec", "aasm_v3_sensitive"]

    def test_all_groups_reference_valid_profiles(self):
        for group_name, profiles in PROFILE_GROUPS.items():
            for p in profiles:
                assert p in PROFILES, (
                    f"Group '{group_name}' references unknown profile '{p}'"
                )

    def test_all_group_contains_every_profile(self):
        """The 'all' group should include every profile in the registry."""
        assert set(PROFILE_GROUPS["all"]) == set(PROFILES.keys())

    def test_aasm_era_has_three(self):
        """AASM era comparison: v1, v2, v3."""
        era = PROFILE_GROUPS["aasm_era"]
        assert len(era) == 3
        assert "aasm_v1_rec" in era
        assert "aasm_v2_rec" in era
        assert "aasm_v3_rec" in era


# ============================================================
# API / helper tests
# ============================================================

class TestPublicAPI:

    def test_list_profiles_returns_sorted(self):
        names = list_profiles()
        assert names == sorted(names)

    def test_list_profiles_by_family(self):
        clinical = list_profiles(family="clinical")
        assert "aasm_v3_rec" in clinical
        assert "mesa_shhs" not in clinical  # dataset, not clinical

        dataset = list_profiles(family="dataset")
        assert "mesa_shhs" in dataset

        legacy = list_profiles(family="legacy")
        assert "chicago_1999" in legacy

        explor = list_profiles(family="exploratory")
        assert "aasm_v3_strict" in explor
        assert "aasm_v3_sensitive" in explor

    def test_get_profile_raises_on_unknown(self):
        with pytest.raises(KeyError):
            get_profile("definitely_does_not_exist")

    def test_profile_metadata_complete(self):
        meta = profile_metadata("aasm_v3_rec")
        assert meta["profile_name"] == "aasm_v3_rec"
        assert meta["flow_threshold_pct"] == 30
        assert meta["desat_threshold_pct"] == 3
        assert meta["desat_or_arousal"] is True
        assert "aasm_version" in meta

    def test_profile_metadata_for_mesa(self):
        # v0.5.0 metadata correction: mesa_shhs now reflects the actual
        # MESA Brigham Reading Center scoring rules (nasal-pressure-primary,
        # 3%-desat-OR-arousal gating = nsrr_ahi_hp3u).
        meta = profile_metadata("mesa_shhs")
        assert meta["profile_family"] == "dataset"
        assert meta["desat_threshold_pct"] == 3

    def test_profile_to_dict_serializable(self):
        """Profile must be JSON-serializable for audit output."""
        import json
        p = get_profile("aasm_v3_rec")
        d = p.to_dict()
        # Should not raise
        json.dumps(d)

    def test_summary_readable(self):
        p = get_profile("aasm_v3_rec")
        s = p.summary()
        assert "30%" in s  # flow threshold
        assert "3%" in s   # desat threshold


# ============================================================
# Integration smoke test
# ============================================================

class TestIntegration:
    """Verify the module can be imported and used as documented."""

    def test_module_imports_clean(self):
        """The primary imports documented in README should work."""
        from psgscoring.profiles import get_profile, list_profiles
        p = get_profile("aasm_v3_rec")
        assert p.hypopnea.flow_reduction_threshold == 0.30

    def test_profile_groups_enumerable(self):
        groups = list_profile_groups()
        assert "clinical" in groups
        assert len(groups["clinical"]) == 3


if __name__ == "__main__":
    import sys
    # Allow running as `python test_profiles.py`
    sys.exit(pytest.main([__file__, "-v"]))
