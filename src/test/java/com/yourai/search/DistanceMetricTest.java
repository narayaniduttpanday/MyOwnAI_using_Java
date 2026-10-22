package com.yourai.search;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class DistanceMetricTest {
    @Test void euclideanDistance() {
        assertEquals(5.0, DistanceMetric.EUCLIDEAN.distance(List.of(0.0,0.0), List.of(3.0,4.0)), 1e-9);
    }
    @Test void manhattanDistance() {
        assertEquals(7.0, DistanceMetric.MANHATTAN.distance(List.of(1.0,2.0), List.of(4.0,6.0)), 1e-9);
    }
    @Test void identicalCosineVectorsHaveZeroDistance() {
        assertEquals(0.0, DistanceMetric.COSINE.distance(List.of(1.0,2.0), List.of(1.0,2.0)), 1e-9);
    }
}
