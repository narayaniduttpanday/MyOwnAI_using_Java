package com.yourai.search;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class SearchIndexTest {
    private static VectorItem item(int id, double... values) {
        List<Double> vector = Arrays.stream(values).boxed().toList();
        return new VectorItem(id, "item-" + id, "test", vector);
    }

    @Test void kdTreeMatchesBruteForceForEuclideanAndManhattan() {
        Random random = new Random(7);
        List<VectorItem> items = new ArrayList<>();
        for (int i = 0; i < 250; i++) {
            items.add(item(i + 1, random.nextDouble() * 20 - 10,
                random.nextDouble() * 20 - 10, random.nextDouble() * 20 - 10));
        }
        KdTreeIndex kd = new KdTreeIndex(items);
        for (DistanceMetric metric : List.of(DistanceMetric.EUCLIDEAN, DistanceMetric.MANHATTAN)) {
            for (int sample = 0; sample < 20; sample++) {
                List<Double> query = List.of(random.nextDouble() * 20 - 10,
                    random.nextDouble() * 20 - 10, random.nextDouble() * 20 - 10);
                List<Integer> expected = BruteForceIndex.search(items, query, 12, metric).stream()
                    .map(n -> n.item().id()).toList();
                List<Integer> actual = kd.search(query, 12, metric).stream().map(n -> n.item().id()).toList();
                assertEquals(expected, actual, "KD-tree should exactly match brute force for " + metric);
            }
        }
    }

    @Test void kdTreeRejectsCosineBecauseSplitPlanePruningIsNotSafe() {
        KdTreeIndex kd = new KdTreeIndex(List.of(item(1, 1, 0), item(2, 0, 1)));
        assertThrows(IllegalArgumentException.class,
            () -> kd.search(List.of(1.0, 0.0), 1, DistanceMetric.COSINE));
    }

    @Test void hnswFindsExactNeighborsOnSmallDataset() {
        List<VectorItem> items = List.of(
            item(1, 0, 0), item(2, 1, 0), item(3, 0, 1), item(4, 2, 2),
            item(5, -1, 0), item(6, 0, -1), item(7, 4, 4), item(8, 0.1, 0.1));
        HnswIndex hnsw = new HnswIndex(items, DistanceMetric.EUCLIDEAN, 4, 50, 50, 17L);
        List<Double> query = List.of(0.05, 0.05);
        List<Integer> expected = BruteForceIndex.search(items, query, 4, DistanceMetric.EUCLIDEAN).stream()
            .map(n -> n.item().id()).toList();
        List<Integer> actual = hnsw.search(query, 4, DistanceMetric.EUCLIDEAN).stream()
            .map(n -> n.item().id()).toList();
        assertEquals(expected, actual);
        assertEquals(items.size(), hnsw.size());
    }

    @Test void hnswRejectsQueriesUsingDifferentMetric() {
        HnswIndex hnsw = new HnswIndex(List.of(item(1, 0, 0), item(2, 1, 1)), DistanceMetric.COSINE);
        assertThrows(IllegalArgumentException.class,
            () -> hnsw.search(List.of(0.5, 0.5), 1, DistanceMetric.EUCLIDEAN));
    }

    @Test void hnswBuildsLayeredGraphAndReportsConnections() {
        List<VectorItem> items = new ArrayList<>();
        for (int i = 0; i < 100; i++) items.add(item(i + 1, i, Math.sin(i), Math.cos(i)));
        HnswIndex hnsw = new HnswIndex(items, DistanceMetric.EUCLIDEAN, 8, 80, 40, 123L);
        Map<String, Object> info = hnsw.graphInfo();
        assertEquals(100, info.get("nodeCount"));
        assertTrue(((List<?>) info.get("edges")).size() > 0);
        assertTrue(((Number) info.get("topLayer")).intValue() >= 0);
    }

    @Test void distanceMetricsRejectNonFiniteValuesAtStoreBoundary() {
        VectorStore store = new VectorStore();
        assertThrows(IllegalArgumentException.class,
            () -> store.insert("bad", "test", Collections.nCopies(16, Double.NaN)));
    }
}
