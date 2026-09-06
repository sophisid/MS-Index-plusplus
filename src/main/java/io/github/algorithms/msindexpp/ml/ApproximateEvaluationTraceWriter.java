package io.github.algorithms.msindexpp.ml;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.utils.Parameters;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;

/** Writes per-query distances needed to analyze target-independent stopping. */
public final class ApproximateEvaluationTraceWriter {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ApproximateEvaluationTraceWriter() {
    }

    public static synchronized void appendQuery(
            String outputPath,
            int queryId,
            int k,
            double targetRecall,
            double achievedRecall,
            boolean stoppedEarly,
            double exactTopKDistance,
            double returnedTopKDistance,
            double firstPassTopKDistance,
            long modelInferenceTimeNs,
            long modelInferenceCalls
    ) throws IOException {
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
        query.put("target_recall", targetRecall);
        query.put("actual_recall", achievedRecall);
        query.put("stopped_early", stoppedEarly);
        query.put("exact_top_k_distance", exactTopKDistance);
        query.put("returned_top_k_distance", returnedTopKDistance);
        query.put("first_pass_top_k_distance", firstPassTopKDistance);
        query.put("model_inference_time_ns", modelInferenceTimeNs);
        query.put("model_inference_calls", modelInferenceCalls);

        try (BufferedWriter writer = Files.newBufferedWriter(
                outputFile.toPath(),
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND,
                StandardOpenOption.CREATE
        )) {
            writer.write(MAPPER.writeValueAsString(query));
            writer.newLine();
        }
    }
}
