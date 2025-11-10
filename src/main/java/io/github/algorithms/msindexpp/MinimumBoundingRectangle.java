package io.github.algorithms.msindexpp;

import lombok.Getter;
import lombok.Setter;
import io.github.utils.LandmarkMBR;
import io.github.utils.rtreemulti.geometry.Rectangle;

/**
 * PP variant of the MinimumBoundingRectangle.
 *
 * Σκοπός: πλήρης ανεξαρτησία στο PP namespace, αλλά συμβατότητα με το shared R-tree API.
 * Κληρονομεί από τον original τύπο ώστε όπου απαιτείται original MBR (π.χ. Geometry.mbr()),
 * να περνάει πολυμορφικά ένα instance αυτής της κλάσης.
 */
public class MinimumBoundingRectangle
        extends io.github.algorithms.msindex.MinimumBoundingRectangle {

    // Πρόσθετα PP fields (δεν επηρεάζουν τον original)
    @Setter @Getter
    private LandmarkMBR landmarkMBR;

    // Προαιρετικό per-query cache μόνο για το PP
    public Double[] distanceCache;

    // --- Constructors ---

    public MinimumBoundingRectangle(double[] mins, double[] maxes) {
        super(mins, maxes);
    }

    public MinimumBoundingRectangle(double[] mins, double[] maxes, LandmarkMBR landmarkMBR) {
        super(mins, maxes);
        this.landmarkMBR = landmarkMBR;
    }

    // --- Factory helpers (όπως είχες πριν) ---

    public static MinimumBoundingRectangle create(double[] mins, double[] maxes) {
        return new MinimumBoundingRectangle(mins, maxes);
    }

    public static MinimumBoundingRectangle create(double[] mins, double[] maxes, LandmarkMBR landmarkMBR) {
        return new MinimumBoundingRectangle(mins, maxes, landmarkMBR);
    }

    // --- Covariant override ---
    // Το original add(Rectangle) επιστρέφει original MBR.
    // Κάνουμε override με πιο συγκεκριμένο return type (PP MBR)
    // ώστε κλήσεις τύπου: runningMBR = runningMBR.add(new MinimumBoundingRectangle(...));
    // να κάνουν compile (τύπος συμβατός με το αριστερό μέλος).
    @Override
    public io.github.algorithms.msindexpp.MinimumBoundingRectangle add(Rectangle r) {
        // Κάνε την πρόσθεση με το original super
        io.github.algorithms.msindex.MinimumBoundingRectangle base = super.add(r);
        // Τύλιξέ το σε PP instance (κρατάς και τυχόν PP metadata αν χρειάζεται)
        return new io.github.algorithms.msindexpp.MinimumBoundingRectangle(
                base.mins(),
                base.maxes(),
                this.landmarkMBR
        );
    }

    // Δεν override-άρουμε mbr()/geometry()/equals()/hashCode() κ.λπ.
    // Τα κληρονομούμε από τον original και παραμένουν συμβατά με το R-tree API.
}
