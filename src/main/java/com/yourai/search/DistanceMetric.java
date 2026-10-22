package com.yourai.search;

import java.util.List;

public enum DistanceMetric {
    COSINE, EUCLIDEAN, MANHATTAN;

    public double distance(List<Double> a, List<Double> b) {
        if (a == null || b == null || a.size() != b.size() || a.isEmpty()) {
            throw new IllegalArgumentException("Vectors must be non-empty and have equal dimensions");
        }
        double sum = 0, dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.size(); i++) {
            double x = a.get(i), y = b.get(i), d = x - y;
            sum += d * d;
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        return switch (this) {
            case EUCLIDEAN -> Math.sqrt(sum);
            case MANHATTAN -> {
                double s = 0;
                for (int i = 0; i < a.size(); i++) s += Math.abs(a.get(i) - b.get(i));
                yield s;
            }
            case COSINE -> (aa < 1e-18 || bb < 1e-18) ? 1.0 : 1.0 - dot / (Math.sqrt(aa) * Math.sqrt(bb));
        };
    }

    public static DistanceMetric parse(String value) {
        try { return valueOf(value == null ? "COSINE" : value.trim().toUpperCase()); }
        catch (IllegalArgumentException ex) { throw new IllegalArgumentException("metric must be cosine, euclidean, or manhattan"); }
    }
}
