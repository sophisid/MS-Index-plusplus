// OODStats.java
package io.github.algorithms.msindexpp;

import static java.lang.Math.*;
import java.util.Arrays;
import lombok.Getter;

public class OODStats implements java.io.Serializable {
    @Getter private final double[] mean;     // length = D
    @Getter private final double[] invStd;   // length = D (1/std)

    private OODStats(double[] mean, double[] invStd) {
        this.mean = mean;
        this.invStd = invStd;
    }

    public static OODStats fromTrails(double[][][] trails, int[] coeffIdx) {
        final int D = coeffIdx.length;
        double[] mean = new double[D];
        double[] var  = new double[D];

        long count = 0;
        for (double[][] ts : trails) {
            for (double[] row : ts) {
                for (int d = 0; d < D; d++) {
                    double v = row[coeffIdx[d]];
                    mean[d] += v;
                }
                count++;
            }
        }
        if (count == 0) return null;
        for (int d = 0; d < D; d++) mean[d] /= count;

        for (double[][] ts : trails) {
            for (double[] row : ts) {
                for (int d = 0; d < D; d++) {
                    double v = row[coeffIdx[d]] - mean[d];
                    var[d] += v * v;
                }
            }
        }
        for (int d = 0; d < D; d++) {
            double std = Math.sqrt(var[d] / Math.max(1, count - 1));
            var[d] = std > 1e-12 ? 1.0 / std : 0.0; // store invStd
        }
        return new OODStats(mean, var);
    }

    /** Zero-alloc: περνάς το ήδη weighted query vector (fftQFlat). */
    public void zScores(double[] q, double[] outMeanMax){ // outMeanMax[0]=zMean, outMeanMax[1]=zMax
        double sumAbs = 0.0, maxAbs = 0.0;
        for (int i = 0; i < q.length; i++) {
            double z = (q[i] - mean[i]) * invStd[i];
            double a = Math.abs(z);
            sumAbs += a;
            if (a > maxAbs) maxAbs = a;
        }
        outMeanMax[0] = sumAbs / q.length;
        outMeanMax[1] = maxAbs;
    }
}
