using System;
using System.Diagnostics;
using UnityEngine;

/// <summary>
/// 4π Uniform Direction Sampling
/// C# Unity version of Python direction_sampler.py
/// 
/// Meeting requirement: "rozkład jednorodny losujemy 4pi"
/// </summary>
public static class DirectionSampler
{
    /// <summary>
    /// Sample single direction uniformly on 4π sphere
    /// </summary>
    public static Vector3 Sample4PiUniform()
    {
        // Sample cos(θ) uniformly in [-1, 1] for uniform solid angle
        float cosTheta = Random.Range(-1f, 1f);
        float theta = Mathf.Acos(cosTheta);

        // Sample φ uniformly in [0, 2π]
        float phi = Random.Range(0f, 2f * Mathf.PI);

        // Convert spherical to Cartesian
        float sinTheta = Mathf.Sin(theta);

        float x = sinTheta * Mathf.Cos(phi);
        float y = sinTheta * Mathf.Sin(phi);
        float z = cosTheta;

        return new Vector3(x, y, z).normalized;
    }

    /// <summary>
    /// Sample multiple directions uniformly on 4π sphere
    /// </summary>
    public static Vector3[] Sample4PiUniform(int numSamples)
    {
        Vector3[] directions = new Vector3[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            directions[i] = Sample4PiUniform();
        }

        return directions;
    }

    /// <summary>
    /// Sample direction in hemisphere defined by normal
    /// </summary>
    public static Vector3 SampleHemisphere(Vector3 normal)
    {
        normal = normal.normalized;

        Vector3 direction;

        do
        {
            direction = Sample4PiUniform();
        }
        while (Vector3.Dot(direction, normal) < 0);

        return direction;
    }

    /// <summary>
    /// Sample direction within cone around forward direction
    /// </summary>
    public static Vector3 SampleCone(Vector3 forwardDirection, float coneAngleDegrees)
    {
        forwardDirection = forwardDirection.normalized;
        float cosAngle = Mathf.Cos(coneAngleDegrees * Mathf.Deg2Rad);

        Vector3 direction;

        do
        {
            direction = Sample4PiUniform();
        }
        while (Vector3.Dot(direction, forwardDirection) < cosAngle);

        return direction;
    }

    /// <summary>
    /// Test uniformity (for debugging)
    /// </summary>
    public static void VisualizeUniformity(int numSamples = 1000)
    {
        Vector3[] directions = Sample4PiUniform(numSamples);

        Debug.Log($"Sampled {numSamples} directions for uniformity test");

        // Check mean (should be ~0)
        Vector3 mean = Vector3.zero;
        foreach (var dir in directions)
        {
            mean += dir;
        }
        mean /= numSamples;

        Debug.Log($"Mean direction (should be ~0): {mean}");
        Debug.Log($"Mean magnitude: {mean.magnitude} (should be close to 0)");
    }
}