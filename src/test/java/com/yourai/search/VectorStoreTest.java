package com.yourai.search;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class VectorStoreTest {
    @SuppressWarnings("unchecked")
    private static List<Integer> ids(VectorStore store, List<Double> query, DistanceMetric metric, String algorithm) {
        Map<String,Object> response = store.search(query, 5, metric, algorithm).get(0);
        return ((List<Map<String,Object>>) response.get("results")).stream()
            .map(row -> ((Number) row.get("id")).intValue()).toList();
    }

    @Test void kdTreeIsExactForSupportedMetricsAndCosineFallsBackSafely() {
        VectorStore store = new VectorStore();
        List<Double> query = store.all().get(3).embedding();
        for (DistanceMetric metric : List.of(DistanceMetric.EUCLIDEAN, DistanceMetric.MANHATTAN)) {
            assertEquals(ids(store, query, metric, "bruteforce"), ids(store, query, metric, "kdtree"));
        }
        Map<String,Object> cosine = store.search(query, 5, DistanceMetric.COSINE, "kdtree").get(0);
        assertEquals("bruteforce-fallback", cosine.get("actualAlgo"));
        assertEquals(ids(store, query, DistanceMetric.COSINE, "bruteforce"), ids(store, query, DistanceMetric.COSINE, "kdtree"));
    }

    @Test void hnswReturnsResultsAndReportsRealGraph() {
        VectorStore store = new VectorStore();
        List<Double> query = store.all().get(0).embedding();
        assertEquals(ids(store, query, DistanceMetric.COSINE, "bruteforce"),
            ids(store, query, DistanceMetric.COSINE, "hnsw"));
        Map<String,Object> graph = store.graphInfo();
        assertEquals(store.size(), graph.get("nodeCount"));
        assertTrue(((List<?>) graph.get("edges")).size() > 0);
    }

    @Test void insertionAndDeletionRebuildSearchIndexes() {
        VectorStore store = new VectorStore();
        List<Double> vector = new ArrayList<>(store.all().get(0).embedding());
        int originalSize = store.size();
        int id = store.insert("test vector", "test", vector).id();
        assertEquals(originalSize + 1, store.graphInfo().get("nodeCount"));
        assertTrue(store.delete(id));
        assertEquals(originalSize, store.graphInfo().get("nodeCount"));
    }

    @Test void benchmarkReportsRecallAndLatency() {
        VectorStore store = new VectorStore();
        Map<String,Object> result = store.benchmark(store.all().get(0).embedding(), 5, DistanceMetric.EUCLIDEAN);
        assertTrue(result.containsKey("bruteforceUs"));
        assertEquals(1.0, result.get("kdtreeRecallAtK"));
        assertTrue(((Number) result.get("hnswRecallAtK")).doubleValue() >= 0.0);
    }
}
