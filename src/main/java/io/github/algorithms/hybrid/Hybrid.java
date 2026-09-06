package io.github.algorithms.hybrid;

import io.github.algorithms.Algorithm;
import io.github.algorithms.AlgorithmType;
import io.github.algorithms.mass.MVMASS;
import io.github.algorithms.msindex.MSIndex;
import io.github.io.DataManager;
import io.github.utils.CandidateMVSubsequence;
import io.github.utils.Parameters;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.logging.Logger;

import static io.github.utils.Parameters.K;
import static io.github.utils.Parameters.N;
import static io.github.utils.Parameters.channels;
import static io.github.utils.Parameters.currQueryId;
import static io.github.utils.Parameters.datasetName;
import static io.github.utils.Parameters.kMeansClusters;
import static io.github.utils.Parameters.maxM;
import static io.github.utils.Parameters.normalize;
import static io.github.utils.Parameters.indexing;
import static io.github.utils.Parameters.parallel;
import static io.github.utils.Parameters.printResults;
import static io.github.utils.Parameters.qLen;
import static io.github.utils.Parameters.queryNoiseEps;
import static io.github.utils.Parameters.queryTime;
import static io.github.utils.Parameters.queryTimePerQuery;
import static io.github.utils.Parameters.seed;
import static io.github.utils.Parameters.selectedVariatesIdx;

public class Hybrid extends Algorithm {
    private static final String TRAINING_CSV_PROPERTY = "hybrid.trainingCsv";
    private static final String EVALUATION_CSV_PROPERTY = "hybrid.evaluationCsv";
    private static final String MODEL_PATH_PROPERTY = "hybrid.modelPath";
    private static final String DEFAULT_ARTIFACTS_DIR_PROPERTY = "hybrid.artifactsDir";
    private static final String DEFAULT_ARTIFACTS_DIR = "artifacts/hybrid";
    private static final int RC_SERIES_SAMPLES = 8;
    private static final int RC_WINDOW_SAMPLES = 16;
    private static final double EPS = 1e-12;

    private final boolean trainingMode;
    private final boolean evaluationMode;
    private final Path trainingCsvPath;
    private final Path evaluationCsvPath;
    private final Path modelPath;

    private MVMASS mass;
    private MSIndex msIndex;
    private QueryDecisionFeatureExtractor featureExtractor;
    private PortableGBDTDecisionModel decisionModel;

    private long evaluationQueries;
    private long correctPredictions;
    private long predictedMassCount;
    private long predictedMsIndexCount;
    private long actualMassCount;
    private long actualMsIndexCount;
    private long totalFeatureExtractionTimeMs;
    private long totalDecisionTimeMs;

    public Hybrid() {
        String artifactsDir = System.getProperty(DEFAULT_ARTIFACTS_DIR_PROPERTY, DEFAULT_ARTIFACTS_DIR).trim();
        String configuredTrainingCsv = System.getProperty(TRAINING_CSV_PROPERTY, "").trim();
        String configuredEvaluationCsv = System.getProperty(EVALUATION_CSV_PROPERTY, "").trim();
        String configuredModelPath = System.getProperty(
                MODEL_PATH_PROPERTY,
                artifactsDir + "/" + datasetName + "/gbdt-model.json"
        ).trim();

        trainingMode = !configuredTrainingCsv.isEmpty();
        evaluationMode = !configuredEvaluationCsv.isEmpty();
        trainingCsvPath = trainingMode ? Path.of(configuredTrainingCsv) : null;
        evaluationCsvPath = evaluationMode ? Path.of(configuredEvaluationCsv) : null;
        modelPath = Path.of(configuredModelPath);
    }

    @Override
    protected void prepareQuerying() {
        if (!trainingMode && Files.exists(modelPath)) {
            decisionModel = PortableGBDTDecisionModel.load(modelPath);
            Logger.getGlobal().info("[Hybrid] Loaded decision model from " + modelPath);
        } else if (!trainingMode) {
            Logger.getGlobal().warning("[Hybrid] No decision model found at " + modelPath + ". Falling back to MSINDEX.");
        }
    }

    @Override
    public String getIndexPath() {
        return "index_caches/" + String.join("_",
                "HYBRID",
                datasetName,
                String.valueOf(N),
                String.valueOf(maxM),
                String.valueOf(seed),
                String.valueOf(qLen),
                String.valueOf(channels),
                normalize ? "1" : "0",
                String.valueOf(kMeansClusters),
                String.valueOf(seed)
        ) + ".marker";
    }

    @Override
    public void buildIndex() {
        Logger logger = Logger.getGlobal();
        mass = new MVMASS();
        msIndex = new MSIndex(true);
        featureExtractor = new QueryDecisionFeatureExtractor();

        logger.info("[Hybrid] Building MASS structures");
        withAlgorithmType(AlgorithmType.MASS, () -> {
            mass.buildIndex();
            return null;
        });

        logger.info("[Hybrid] Building MS-Index structures");
        withAlgorithmType(AlgorithmType.MSINDEX, () -> {
            msIndex.buildIndex();
            return null;
        });
    }

    @Override
    public void saveIndex() {
        withAlgorithmType(AlgorithmType.MASS, () -> {
            mass.saveIndex();
            return null;
        });
        withAlgorithmType(AlgorithmType.MSINDEX, () -> {
            msIndex.saveIndex();
            return null;
        });
        try {
            Path markerPath = Path.of(getIndexPath());
            if (markerPath.getParent() != null) {
                Files.createDirectories(markerPath.getParent());
            }
            Files.writeString(markerPath, "hybrid\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write hybrid index marker", e);
        }
    }

    @Override
    public Algorithm loadIndex() {
        mass = new MVMASS();
        msIndex = new MSIndex(true);
        featureExtractor = new QueryDecisionFeatureExtractor();

        MVMASS loadedMass = withAlgorithmType(AlgorithmType.MASS, () -> (MVMASS) mass.loadIndex());
        MSIndex loadedMsIndex = withAlgorithmType(AlgorithmType.MSINDEX, () -> (MSIndex) msIndex.loadIndex());
        if (loadedMass != null) {
            mass = loadedMass;
        }
        if (loadedMsIndex != null) {
            msIndex = loadedMsIndex;
        }
        return this;
    }

    @Override
    public void queryAll() {
        Logger logger = Logger.getGlobal();
        logger.info("Starting querying process");

        indexing = false;
        if (!parallel) {
            ConcurrencyUtils.setNumberOfThreads(1);
        }

        double[][][] queries = DataManager.queries;
        queryTimePerQuery = new long[queries.length];
        prepareQuerying();

        for (int i = 0; i < queries.length; i++) {
            double[][] query = queries[i];
            long start = System.currentTimeMillis();
            long featureStart = System.currentTimeMillis();
            DecisionFeatureVector featureVector = featureExtractor.extract(query);
            long featureElapsed = System.currentTimeMillis() - featureStart;
            totalFeatureExtractionTimeMs += featureElapsed;

            logger.info(String.format("Querying %d/%d", i, queries.length));

            QueryExecution execution = trainingMode
                    ? runTrainingQuery(i, query, featureVector)
                    : (evaluationMode ? runEvaluatedQuery(i, query, featureVector, featureElapsed) : runPredictedQuery(query, featureVector));
            long elapsed = System.currentTimeMillis() - start;

            if (printResults) {
                execution.results.sort(Comparator.comparingDouble(CandidateMVSubsequence::totalDistance));
                logger.info(String.format("Results for query %d/%d: ", i, queries.length) + execution.results);
            }

            queryTimePerQuery[i] = elapsed;
            queryTime += elapsed;
            currQueryId++;
        }

        if (evaluationMode && evaluationQueries > 0) {
            logger.info(String.format("[Hybrid] Decision accuracy: %.4f (%d/%d)",
                    correctPredictions / (double) evaluationQueries,
                    correctPredictions,
                    evaluationQueries));
            logger.info(String.format("[Hybrid] Predicted counts: MASS=%d, MSINDEX=%d", predictedMassCount, predictedMsIndexCount));
            logger.info(String.format("[Hybrid] Actual best counts: MASS=%d, MSINDEX=%d", actualMassCount, actualMsIndexCount));
        }

        if (queries.length > 0) {
            logger.info(String.format("[Hybrid] Avg feature extraction time: %.4fms",
                    totalFeatureExtractionTimeMs / (double) queries.length));
            logger.info(String.format("[Hybrid] Avg decision time: %.4fms",
                    totalDecisionTimeMs / (double) queries.length));
        }
    }

    @Override
    public List<CandidateMVSubsequence> kNN(int k, double[][] query) {
        DecisionFeatureVector featureVector = featureExtractor.extract(query);
        return runPredictedQuery(query, featureVector).results;
    }

    @Override
    public double memoryUsage() {
        double massMemory = mass == null ? 0d : mass.memoryUsage();
        double msIndexMemory = msIndex == null ? 0d : msIndex.memoryUsage();
        return massMemory + msIndexMemory;
    }

    private QueryExecution runTrainingQuery(int queryId, double[][] query, DecisionFeatureVector featureVector) {
        TimedResult massResult = timeQuery(() -> mass.kNN(K, query));
        TimedResult msIndexResult = timeQuery(() -> msIndex.kNN(K, query));

        DecisionTrainingCsvWriter.append(
                trainingCsvPath,
                queryId,
                featureVector,
                massResult.elapsedMs,
                msIndexResult.elapsedMs
        );

        boolean massIsBetter = massResult.elapsedMs < msIndexResult.elapsedMs;
        Logger.getGlobal().info(String.format(
                "[Hybrid] Training query %d: MASS=%dms, MSINDEX=%dms, winner=%s",
                queryId,
                massResult.elapsedMs,
                msIndexResult.elapsedMs,
                massIsBetter ? AlgorithmType.MASS : AlgorithmType.MSINDEX
        ));

        return new QueryExecution(massIsBetter ? massResult.results : msIndexResult.results);
    }

    private QueryExecution runPredictedQuery(double[][] query, DecisionFeatureVector featureVector) {
        long decisionStart = System.nanoTime();
        AlgorithmType selectedAlgorithm = predictWithTiming(featureVector);
        long decisionElapsed = (System.nanoTime() - decisionStart) / 1_000_000L;
        totalDecisionTimeMs += decisionElapsed;

        Logger.getGlobal().info("[Hybrid] Decision time: " + decisionElapsed + "ms");
        Logger.getGlobal().info("[Hybrid] Selected " + selectedAlgorithm + " for current query");

        if (selectedAlgorithm == AlgorithmType.MASS) {
            return new QueryExecution(mass.kNN(K, query));
        }
        return new QueryExecution(msIndex.kNN(K, query));
    }

    private QueryExecution runEvaluatedQuery(
            int queryId,
            double[][] query,
            DecisionFeatureVector featureVector,
            long featureElapsed
    ) {
        double rcEstimate = estimateRCViaSamples(query);
        long decisionStart = System.nanoTime();
        AlgorithmType predictedAlgorithm = predictWithTiming(featureVector);
        long decisionElapsed = (System.nanoTime() - decisionStart) / 1_000_000L;
        totalDecisionTimeMs += decisionElapsed;

        TimedResult massResult = timeQuery(() -> mass.kNN(K, query));
        TimedResult msIndexResult = timeQuery(() -> msIndex.kNN(K, query));

        AlgorithmType actualBestAlgorithm = massResult.elapsedMs < msIndexResult.elapsedMs
                ? AlgorithmType.MASS
                : AlgorithmType.MSINDEX;
        TimedResult predictedResult = predictedAlgorithm == AlgorithmType.MASS ? massResult : msIndexResult;
        long hybridTotalTimeMs = featureElapsed + decisionElapsed + predictedResult.elapsedMs;
        boolean predictionCorrect = predictedAlgorithm == actualBestAlgorithm;

        evaluationQueries++;
        if (predictionCorrect) {
            correctPredictions++;
        }
        if (predictedAlgorithm == AlgorithmType.MASS) {
            predictedMassCount++;
        } else {
            predictedMsIndexCount++;
        }
        if (actualBestAlgorithm == AlgorithmType.MASS) {
            actualMassCount++;
        } else {
            actualMsIndexCount++;
        }

        DecisionEvaluationCsvWriter.append(
                evaluationCsvPath,
                queryId,
                featureVector,
                rcEstimate,
                predictedAlgorithm,
                actualBestAlgorithm,
                featureElapsed,
                decisionElapsed,
                hybridTotalTimeMs,
                predictedResult.elapsedMs,
                massResult.elapsedMs,
                msIndexResult.elapsedMs
        );

        Logger.getGlobal().info(String.format(
                "[Hybrid] Eval query %d: rc=%.5f predicted=%s actual=%s feature=%dms decision=%dms hybridTotal=%dms MASS=%dms MSINDEX=%dms",
                queryId,
                rcEstimate,
                predictedAlgorithm,
                actualBestAlgorithm,
                featureElapsed,
                decisionElapsed,
                hybridTotalTimeMs,
                massResult.elapsedMs,
                msIndexResult.elapsedMs
        ));

        return new QueryExecution(predictedResult.results);
    }

    private AlgorithmType predictWithTiming(DecisionFeatureVector featureVector) {
        if (decisionModel == null) {
            return AlgorithmType.MSINDEX;
        }
        long start = System.nanoTime();
        try {
            return decisionModel.predict(featureVector);
        } finally {
            Parameters.hybridModelInferenceTimeNs += System.nanoTime() - start;
            Parameters.hybridModelInferenceCalls++;
        }
    }

    private static TimedResult timeQuery(QueryRunner runner) {
        long start = System.currentTimeMillis();
        List<CandidateMVSubsequence> results = runner.run();
        return new TimedResult(results, System.currentTimeMillis() - start);
    }

    private <T> T withAlgorithmType(AlgorithmType targetType, Callable<T> action) {
        AlgorithmType previousType = io.github.utils.Parameters.algorithmType;
        io.github.utils.Parameters.algorithmType = targetType;
        try {
            return action.call();
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            io.github.utils.Parameters.algorithmType = previousType;
        }
    }

    private interface QueryRunner {
        List<CandidateMVSubsequence> run();
    }

    private double estimateRCViaSamples(double[][] query) {
        double[][][] data = DataManager.data;
        if (data.length == 0) {
            return 1.0;
        }

        double[] sampledDistances = new double[RC_SERIES_SAMPLES * RC_WINDOW_SAMPLES];
        double distanceSum = 0.0;
        int sampledCount = 0;

        for (int s = 0; s < RC_SERIES_SAMPLES; s++) {
            int tsId = Parameters.random.nextInt(data.length);
            int seriesLength = DataManager.getM(tsId);
            int limit = Math.max(1, seriesLength - qLen + 1);

            for (int w = 0; w < RC_WINDOW_SAMPLES; w++) {
                int start = Parameters.random.nextInt(limit);
                double dist2 = 0.0;
                for (int ch : selectedVariatesIdx) {
                    double distance = windowDistance(data[tsId][ch], start, query[ch], normalize);
                    dist2 += distance * distance;
                }
                double distance = Math.sqrt(dist2);
                sampledDistances[sampledCount++] = distance;
                distanceSum += distance;
            }
        }

        if (sampledCount == 0 || !Double.isFinite(distanceSum)) {
            return 1.0;
        }

        Arrays.sort(sampledDistances, 0, sampledCount);
        int rank = Math.min(Math.max(K, 1), sampledCount) - 1;
        double kthNearestDistance = sampledDistances[rank];
        double meanDistance = distanceSum / sampledCount;
        return meanDistance / Math.max(kthNearestDistance, EPS);
    }

    private double windowDistance(double[] series, int start, double[] query, boolean zNormalize) {
        if (!zNormalize) {
            double sum = 0.0;
            for (int i = 0; i < qLen; i++) {
                double diff = series[start + i] - query[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }

        double muSeries = 0.0;
        double muQuery = 0.0;
        for (int i = 0; i < qLen; i++) {
            muSeries += series[start + i];
            muQuery += query[i];
        }
        muSeries /= qLen;
        muQuery /= qLen;

        double varSeries = 0.0;
        double varQuery = 0.0;
        for (int i = 0; i < qLen; i++) {
            double ds = series[start + i] - muSeries;
            double dq = query[i] - muQuery;
            varSeries += ds * ds;
            varQuery += dq * dq;
        }
        double stdSeries = Math.sqrt(Math.max(varSeries, 0.0) + EPS);
        double stdQuery = Math.sqrt(Math.max(varQuery, 0.0) + EPS);

        double sum = 0.0;
        for (int i = 0; i < qLen; i++) {
            double zs = (series[start + i] - muSeries) / stdSeries;
            double zq = (query[i] - muQuery) / stdQuery;
            double diff = zs - zq;
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private static final class TimedResult {
        private final List<CandidateMVSubsequence> results;
        private final long elapsedMs;

        private TimedResult(List<CandidateMVSubsequence> results, long elapsedMs) {
            this.results = results;
            this.elapsedMs = elapsedMs;
        }
    }

    private static final class QueryExecution {
        private final List<CandidateMVSubsequence> results;

        private QueryExecution(List<CandidateMVSubsequence> results) {
            this.results = results;
        }
    }
}
