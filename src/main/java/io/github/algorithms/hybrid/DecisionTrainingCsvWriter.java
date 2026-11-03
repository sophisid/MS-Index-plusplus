package io.github.algorithms.hybrid;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

public final class DecisionTrainingCsvWriter {
    private static final String[] EXTRA_COLUMNS = {
            "query_id",
            "mass_time_ms",
            "msindex_time_ms",
            "better_algorithm",
            "mass_is_better"
    };

    private DecisionTrainingCsvWriter() {
    }

    public static synchronized void append(
            Path path,
            int queryId,
            DecisionFeatureVector featureVector,
            long massTimeMs,
            long msIndexTimeMs
    ) {
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            if (!Files.exists(path)) {
                Files.writeString(path, header() + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
            }

            List<String> row = new ArrayList<>();
            row.add(Integer.toString(queryId));
            featureVector.asMap().values().forEach(value -> row.add(Double.toString(value)));
            row.add(Long.toString(massTimeMs));
            row.add(Long.toString(msIndexTimeMs));
            row.add(massTimeMs < msIndexTimeMs ? "MASS" : "MSINDEX");
            row.add(massTimeMs < msIndexTimeMs ? "1" : "0");

            Files.writeString(path, String.join(",", row) + "\n", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new RuntimeException("Failed to append hybrid training row to " + path, e);
        }
    }

    private static String header() {
        List<String> columns = new ArrayList<>();
        columns.add(EXTRA_COLUMNS[0]);
        for (String featureName : DecisionFeatureVector.FEATURE_NAMES) {
            columns.add(featureName);
        }
        for (int i = 1; i < EXTRA_COLUMNS.length; i++) {
            columns.add(EXTRA_COLUMNS[i]);
        }
        return String.join(",", columns);
    }
}
