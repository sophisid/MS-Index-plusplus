package io.github.algorithms.msindexpp.decision;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class DecisionFeatureVector {
    public static final String[] FEATURE_NAMES = {
            "k",
            "dataset_channels",
            "query_channels",
            "dataset_series_count",
            "avg_series_length",
            "std_series_length",
            "avg_subsequences_per_series",
            "total_subsequences",
            "query_length",
            "query_length_ratio",
            "query_mu_mean",
            "query_sigma2_mean",
            "query_abs_mean_mean",
            "query_std_mean",
            "query_std_std",
            "query_energy_mean",
            "query_energy_std"
    };

    private final double[] values;

    public DecisionFeatureVector(double[] values) {
        if (values.length != FEATURE_NAMES.length) {
            throw new IllegalArgumentException("Expected " + FEATURE_NAMES.length + " features but got " + values.length);
        }
        this.values = Arrays.copyOf(values, values.length);
    }

    public double[] values() {
        return Arrays.copyOf(values, values.length);
    }

    public Map<String, Double> asMap() {
        LinkedHashMap<String, Double> map = new LinkedHashMap<>();
        for (int i = 0; i < FEATURE_NAMES.length; i++) {
            map.put(FEATURE_NAMES[i], values[i]);
        }
        return map;
    }
}
