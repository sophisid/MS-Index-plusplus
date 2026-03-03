package io.github.utils;

public class FourierTrail {
    private double[][] trail;

    private LandmarkPortfolio[] landmarkPortfolios;

    public FourierTrail(double[][] trail, LandmarkPortfolio[] landmarkPortfolios) {
        this.trail = trail;
        this.landmarkPortfolios = landmarkPortfolios;
    }

    public int getLength() {
        return trail.length;
    }

    public double[][] getTrail() {
        return trail;
    }

    public void setTrail(double[][] trail) {
        this.trail = trail;
    }

    public LandmarkPortfolio[] getLandmarkPortfolios() {
        return landmarkPortfolios;
    }

    public void setLandmarkPortfolios(LandmarkPortfolio[] landmarkPortfolios) {
        this.landmarkPortfolios = landmarkPortfolios;
    }
}
