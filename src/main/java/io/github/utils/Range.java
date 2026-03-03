package io.github.utils;

import java.io.Serializable;
import java.util.List;

import static io.github.utils.Parameters.qLen;

public class Range implements Serializable {
    private int start;

    private int end;

    public Range(int start, int end) {
        this.start = start;
        this.end = end;
    }

    public int getStart() {
        return start;
    }

    public void setStart(int start) {
        this.start = start;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    public Tuple2<List<Range>, Double> optimalPlan;

    public int getPower2Length() {
        return lib.nextPowerOfTwo(end - start + qLen);
    }
    public int getFullLength() {
        return end - start + qLen;
    }

    public int getLength() {
        return end - start + 1;
    }

    public void reset(){optimalPlan = null;}

    public int compareTo(Range range) {
        return Integer.compare(this.start, range.start);
    }

    public String toString() {
        return String.format("[%d, %d]", start, end);
    }

    public boolean overlaps(Range other) {
        return this.start <= other.end && other.start <= this.end;
    }

    public boolean neighbours(Range other) {return this.end == other.start - 1 || this.start == other.end + 1;}

    public Range union(Range other) {
        return new Range(Math.min(this.start, other.start), Math.max(this.end, other.end));
    }

    public Range intersection(Range other) {
        return new Range(Math.max(this.start, other.start), Math.min(this.end, other.end));
    }
}
