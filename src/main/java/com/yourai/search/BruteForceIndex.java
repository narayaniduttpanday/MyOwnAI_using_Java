package com.yourai.search;

import java.util.*;

/** Exact nearest-neighbour baseline; useful for correctness checks and recall measurement. */
public final class BruteForceIndex {
    private BruteForceIndex() {}

    public static List<Neighbor> search(Collection<VectorItem> items, List<Double> query,
                                        int k, DistanceMetric metric) {
        return items.stream()
            .map(item -> new Neighbor(item, metric.distance(query, item.embedding())))
            .sorted(Comparator.comparingDouble(Neighbor::distance).thenComparingInt(n -> n.item().id()))
            .limit(k)
            .toList();
    }

    public record Neighbor(VectorItem item, double distance) {}
}
