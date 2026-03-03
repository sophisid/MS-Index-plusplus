package io.github.algorithms.msindexshared;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public final class BoundedPriorityQueue<T> {

    private final PriorityQueue<T> queue;
    private final Comparator<? super T> comparator;
    private final int maxSize;

    public BoundedPriorityQueue(final int maxSize, final Comparator<? super T> comparator) {
        this.queue = new PriorityQueue<>(reverse(comparator));
        this.comparator = comparator;
        this.maxSize = maxSize;
    }

    private static <T> Comparator<T> reverse(final Comparator<T> comparator) {
        return (o1, o2) -> comparator.compare(o2, o1);
    }

    public static <T> BoundedPriorityQueue<T> create(final int maxSize,
                                                     final Comparator<? super T> comparator) {
        return new BoundedPriorityQueue<>(maxSize, comparator);
    }

    public void add(final T t) {
        if (t == null) {
            throw new NullPointerException("cannot add null to the queue");
        }
        while (queue.size() >= maxSize) {
            final T maxElement = queue.peek();
            int compInt = comparator.compare(maxElement, t);
            if (compInt < 0) {
                return;
            } else if (compInt > 0) {
                queue.poll();
            } else {
                queue.add(t);
                return;
            }
        }
        queue.add(t);
    }

    public void addAll(final List<T> list) {
        for (T t : list) {
            add(t);
        }
    }

    public T peek() {
        return queue.peek();
    }

    public int size() {
        return queue.size();
    }

    public List<T> asOrderedList() {
        List<T> list = new ArrayList<>(queue);
        list.sort(comparator);
        return list;
    }
}
