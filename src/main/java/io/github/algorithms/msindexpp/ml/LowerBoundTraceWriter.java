package io.github.algorithms.msindexpp.ml;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.algorithms.msindexshared.MinimumBoundingRectangle;
import io.github.utils.Parameters;
import io.github.utils.rtreemulti.Entry;
import io.github.utils.rtreemulti.RTree;
import io.github.utils.rtreemulti.geometry.Geometry;
import io.github.utils.rtreemulti.internal.LeafDefault;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/** Writes the complete leaf-entry lower-bound distribution for one query. */
public final class LowerBoundTraceWriter {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private LowerBoundTraceWriter() {
    }

    public static void appendQuery(String outputPath, int queryId, int k, double threshold,
                                   MinimumBoundingRectangle queryMBR,
                                   RTree<?, Geometry> tree) throws IOException {
        appendQuery(outputPath, queryId, k, threshold, Double.NaN, queryMBR, tree);
    }

    public static void appendQuery(String outputPath, int queryId, int k, double threshold,
                                   double relativeContrast, MinimumBoundingRectangle queryMBR,
                                   RTree<?, Geometry> tree) throws IOException {
        appendQuery(outputPath, queryId, k, threshold,
                Double.isFinite(relativeContrast) ? Map.of(k, relativeContrast) : null,
                queryMBR, tree);
    }

    public static void appendQuery(String outputPath, int queryId, int k, double threshold,
                                   Map<Integer, Double> relativeContrasts,
                                   MinimumBoundingRectangle queryMBR,
                                   RTree<?, Geometry> tree) throws IOException {
        File outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        ObjectNode query = MAPPER.createObjectNode();
        query.put("dataset", Parameters.datasetName);
        query.put("query_id", queryId);
        query.put("requested_k", k);
        query.put("normalize", Parameters.normalize);
        query.put("query_noise_eps", Parameters.queryNoiseEps);
        query.put("threshold", threshold);
        if (relativeContrasts != null) {
            Double relativeContrast = relativeContrasts.get(k);
            if (relativeContrast != null && Double.isFinite(relativeContrast)) {
                query.put("relative_contrast", relativeContrast);
            }
            if (!relativeContrasts.isEmpty()) {
                ObjectNode rcByK = query.putObject("relative_contrast_by_k");
                for (Map.Entry<Integer, Double> entry : relativeContrasts.entrySet()) {
                    if (entry.getValue() != null && Double.isFinite(entry.getValue())) {
                        rcByK.put(String.valueOf(entry.getKey()), entry.getValue());
                    }
                }
            }
        }

        ArrayNode lowerBounds = query.putArray("lower_bounds");
        long count = 0;
        for (LeafDefault<?, Geometry> leaf : tree.leafs()) {
            for (Entry<?, Geometry> entry : leaf.entries()) {
                lowerBounds.add(entry.geometry().distance(queryMBR));
                count++;
            }
        }
        query.put("n_entries", count);

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile.toPath(), StandardCharsets.UTF_8,
                StandardOpenOption.APPEND, StandardOpenOption.CREATE)) {
            writer.write(MAPPER.writeValueAsString(query));
            writer.newLine();
        }
    }
}
