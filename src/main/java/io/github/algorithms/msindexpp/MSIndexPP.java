package io.github.algorithms.msindexpp;

import io.github.algorithms.Algorithm;
import io.github.algorithms.AlgorithmType;
import io.github.algorithms.mass.MVMASS;
import io.github.algorithms.msindex.MSIndex;
import io.github.algorithms.msindexpp.decision.DecisionEvaluationCsvWriter;
import io.github.algorithms.msindexpp.decision.DecisionFeatureVector;
import io.github.algorithms.msindexpp.decision.DecisionTrainingCsvWriter;
import io.github.algorithms.msindexpp.decision.GradientBoostedDecisionModel;
import io.github.algorithms.msindexpp.decision.QueryDecisionFeatureExtractor;
import io.github.io.DataManager;
import io.github.utils.CandidateMVSubsequence;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.logging.Logger;

import static io.github.utils.Parameters.K;
import static io.github.utils.Parameters.currQueryId;
import static io.github.utils.Parameters.indexing;
import static io.github.utils.Parameters.parallel;
import static io.github.utils.Parameters.printResults;
import static io.github.utils.Parameters.queryTime;
import static io.github.utils.Parameters.queryTimePerQuery;

public class MSIndexPP extends Algorithm {
    private static final String TRAINING_CSV_PROPERTY = "msindexpp.trainingCsv";
    private static final String EVALUATION_CSV_PROPERTY = "msindexpp.evaluationCsv";
    private static final String MODEL_PATH_PROPERTY = "msindexpp.modelPath";
    private static final String DEFAULT_MODEL_PATH = "output/msindexpp_gbdt_model.json";

    private final boolean trainingMode;
    private final Path trainingCsvPath;
    private final boolean evaluationMode;
    private final Path evaluationCsvPath;
    private final Path modelPath;

    private MSIndex msIndex;
    private MVMASS mass;
    private QueryDecisionFeatureExtractor featureExtractor;
    private GradientBoostedDecisionModel decisionModel;
    private long evaluationQueries;
    private long correctPredictions;
    private long predictedMassCount;
    private long predictedMsIndexCount;
    private long actualMassCount;
    private long actualMsIndexCount;
    private long totalFeatureExtractionTimeMs;
    private long totalDecisionTimeMs;

    public MSIndexPP(boolean ignoredBuildTree) {
        final String trainingCsv = System.getProperty(TRAINING_CSV_PROPERTY, "").trim();
        final String evaluationCsv = System.getProperty(EVALUATION_CSV_PROPERTY, "").trim();
        final String configuredModelPath = System.getProperty(MODEL_PATH_PROPERTY, DEFAULT_MODEL_PATH).trim();
        trainingMode = !trainingCsv.isEmpty();
        trainingCsvPath = trainingMode ? Path.of(trainingCsv) : null;
        evaluationMode = !evaluationCsv.isEmpty();
        evaluationCsvPath = evaluationMode ? Path.of(evaluationCsv) : null;
        modelPath = Path.of(configuredModelPath);
    }

    @Override
    public void buildIndex() {
        Logger logger = Logger.getGlobal();
        mass = new MVMASS();
        msIndex = new MSIndex(true);
        featureExtractor = new QueryDecisionFeatureExtractor();

        logger.info("[MS-IndexPP] Building MASS structures");
        mass.buildIndex();

        logger.info("[MS-IndexPP] Building MS-Index structures");
        withAlgorithmType(AlgorithmType.MSINDEX, () -> {
            msIndex.buildIndex();
            return null;
        });

        if (trainingMode) {
            logger.info("[MS-IndexPP] Training mode enabled. Writing rows to " + trainingCsvPath);
            return;
        }

        if (Files.exists(modelPath)) {
            decisionModel = GradientBoostedDecisionModel.load(modelPath);
            logger.info("[MS-IndexPP] Loaded decision model from " + modelPath);
        } else {
            logger.warning("[MS-IndexPP] No decision model found at " + modelPath + ". Falling back to MS-Index.");
        }
    }

    @Override
    public void queryAll() {
        Logger logger = Logger.getGlobal();
        logger.info("Starting querying process");

        indexing = false;
        if (!parallel) {
            ConcurrencyUtils.setNumberOfThreads(1);
        }

        final double[][][] queries = DataManager.queries;
        queryTimePerQuery = new long[queries.length];

        for (int i = 0; i < queries.length; i++) {
            final double[][] query = queries[i];
            final long featureStart = System.currentTimeMillis();
            final DecisionFeatureVector featureVector = featureExtractor.extract(query);
            final long featureElapsed = System.currentTimeMillis() - featureStart;
            totalFeatureExtractionTimeMs += featureElapsed;

            logger.info(String.format("Querying %d/%d", i, queries.length));
            logger.info(String.format("[MS-IndexPP] Feature extraction time: %dms", featureElapsed));

            final long start = System.currentTimeMillis();
            final QueryExecution execution = trainingMode
                    ? runTrainingQuery(i, query, featureVector)
                    : (evaluationMode
                    ? runEvaluatedQuery(i, query, featureVector)
                    : runPredictedQuery(query, featureVector));
            final long elapsed = System.currentTimeMillis() - start;

            if (printResults) {
                execution.results.sort(Comparator.comparingDouble(CandidateMVSubsequence::totalDistance));
                logger.info(String.format("Results for query %d/%d: ", i, queries.length) + execution.results);
            }

            queryTimePerQuery[i] = elapsed;
            queryTime += elapsed;
            currQueryId++;
        }

        if (evaluationMode && evaluationQueries > 0) {
            logger.info(String.format(
                    "[MS-IndexPP] Decision accuracy: %.4f (%d/%d)",
                    correctPredictions / (double) evaluationQueries,
                    correctPredictions,
                    evaluationQueries
            ));
            logger.info(String.format(
                    "[MS-IndexPP] Predicted counts: MASS=%d, MSINDEX=%d",
                    predictedMassCount,
                    predictedMsIndexCount
            ));
            logger.info(String.format(
                    "[MS-IndexPP] Actual best counts: MASS=%d, MSINDEX=%d",
                    actualMassCount,
                    actualMsIndexCount
            ));
        }

        if (queries.length > 0) {
            logger.info(String.format(
                    "[MS-IndexPP] Avg feature extraction time: %.4fms",
                    totalFeatureExtractionTimeMs / (double) queries.length
            ));
            logger.info(String.format(
                    "[MS-IndexPP] Avg decision time: %.4fms",
                    totalDecisionTimeMs / (double) queries.length
            ));
        }
    }

    @Override
    public List<CandidateMVSubsequence> kNN(int k, double[][] query) {
        final DecisionFeatureVector featureVector = featureExtractor.extract(query);
        return runPredictedQuery(query, featureVector).results;
    }

    @Override
    public double memoryUsage() {
        double massMemory = mass == null ? 0 : mass.memoryUsage();
        double msIndexMemory = msIndex == null ? 0 : msIndex.memoryUsage();
        return massMemory + msIndexMemory;
    }

    private QueryExecution runTrainingQuery(int queryId, double[][] query, DecisionFeatureVector featureVector) {
        final TimedResult massResult = timeQuery(() -> mass.kNN(K, query));
        final TimedResult msIndexResult = timeQuery(() -> msIndex.kNN(K, query));

        final boolean massIsBetter = massResult.elapsedMs < msIndexResult.elapsedMs;
        DecisionTrainingCsvWriter.append(
                trainingCsvPath,
                queryId,
                featureVector,
                massResult.elapsedMs,
                msIndexResult.elapsedMs
        );

        Logger.getGlobal().info(String.format(
                "[MS-IndexPP] Training query %d: MASS=%dms, MSINDEX=%dms, winner=%s",
                queryId,
                massResult.elapsedMs,
                msIndexResult.elapsedMs,
                massIsBetter ? AlgorithmType.MASS : AlgorithmType.MSINDEX
        ));

        return new QueryExecution(massIsBetter ? massResult.results : msIndexResult.results);
    }

    private QueryExecution runPredictedQuery(double[][] query, DecisionFeatureVector featureVector) {
        final long decisionStart = System.currentTimeMillis();
        final AlgorithmType selectedAlgorithm = decisionModel == null
                ? AlgorithmType.MSINDEX
                : decisionModel.predict(featureVector);
        final long decisionElapsed = System.currentTimeMillis() - decisionStart;
        totalDecisionTimeMs += decisionElapsed;

        Logger.getGlobal().info("[MS-IndexPP] Decision time: " + decisionElapsed + "ms");
        Logger.getGlobal().info("[MS-IndexPP] Selected " + selectedAlgorithm + " for current query");

        if (selectedAlgorithm == AlgorithmType.MASS) {
            return new QueryExecution(mass.kNN(K, query));
        }
        return new QueryExecution(msIndex.kNN(K, query));
    }

    private QueryExecution runEvaluatedQuery(int queryId, double[][] query, DecisionFeatureVector featureVector) {
        final long decisionStart = System.currentTimeMillis();
        final AlgorithmType predictedAlgorithm = decisionModel == null
                ? AlgorithmType.MSINDEX
                : decisionModel.predict(featureVector);
        final long decisionElapsed = System.currentTimeMillis() - decisionStart;
        totalDecisionTimeMs += decisionElapsed;

        final TimedResult massResult = timeQuery(() -> mass.kNN(K, query));
        final TimedResult msIndexResult = timeQuery(() -> msIndex.kNN(K, query));

        final AlgorithmType actualBestAlgorithm = massResult.elapsedMs < msIndexResult.elapsedMs
                ? AlgorithmType.MASS
                : AlgorithmType.MSINDEX;
        final TimedResult predictedResult = predictedAlgorithm == AlgorithmType.MASS ? massResult : msIndexResult;
        final boolean predictionCorrect = predictedAlgorithm == actualBestAlgorithm;

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
                predictedAlgorithm,
                actualBestAlgorithm,
                predictedResult.elapsedMs,
                massResult.elapsedMs,
                msIndexResult.elapsedMs
        );

        Logger.getGlobal().info(String.format(
                "[MS-IndexPP] Decision time=%dms, predicted=%s, actualBest=%s, correct=%s, MASS=%dms, MSINDEX=%dms",
                decisionElapsed,
                predictedAlgorithm,
                actualBestAlgorithm,
                predictionCorrect,
                massResult.elapsedMs,
                msIndexResult.elapsedMs
        ));

        return new QueryExecution(predictedResult.results);
    }

    private TimedResult timeQuery(QueryRunner queryRunner) {
        final long start = System.currentTimeMillis();
        final List<CandidateMVSubsequence> results = queryRunner.run();
        return new TimedResult(results, System.currentTimeMillis() - start);
    }

    private static <T> T withAlgorithmType(AlgorithmType targetType, AlgorithmSupplier<T> supplier) {
        final AlgorithmType previousType = io.github.utils.Parameters.algorithmType;
        io.github.utils.Parameters.algorithmType = targetType;
        try {
            return supplier.get();
        } finally {
            io.github.utils.Parameters.algorithmType = previousType;
        }
    }

    @FunctionalInterface
    private interface QueryRunner {
        List<CandidateMVSubsequence> run();
    }

    @FunctionalInterface
    private interface AlgorithmSupplier<T> {
        T get();
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
