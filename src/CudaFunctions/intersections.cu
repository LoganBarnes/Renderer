#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h" // includes helper_math.h
#include "shared/renderObjects.hpp"
#include "intersections.cuh"

__constant__ int INF = 1000;
__constant__ int EPS = 0.001;

extern "C"
{

    __device__
    int solveQuadratic(float a, float b, float c, float &t1, float &t2)
    {
        float discriminant = b * b - 4.0 * a * c;

        // Discriminant is 0. One solution exists.
        if (abs(discriminant) < EPS) // epsilon value
        {
            if (abs(b) < EPS)
                return 0;

            if (abs(a) < EPS)
            {
                t1 = INF;
                return 1;
            }

            // else
            t1 = -b / (2.0 * a);
            return 1;
        }

        // Discriminant is less than 0. No solutions exists.
        if (discriminant < 0.0)
            return 0;

        // Discriminant is greater than 0. Two solutions exists.
        // if (abs(a) < EPS)
        // {
        // 	t1 = INF;
        // 	t2 = -INF;
        // 	return 2;
        // }
        // else
        {
            float sqrtDisc = sqrt(discriminant);
            t1 = (-b + sqrtDisc) / (2.0 * a);
            t2 = (-b - sqrtDisc) / (2.0 * a);
        }
        return 2;
    }

    __device__
    float4 intersectSphere(float3 E, float3 D)
    {
        float4 n = make_float4(0, 0, 0, INF);
        float3 p;

        float a = dot(D, D);
        float b = 2.0 * dot(D, E);
        float c = dot(E, E) - 1.0;

        float t1, t2;
        int solutions = solveQuadratic(a, b, c, t1, t2);

        if (solutions > 0)
        {
            if (t1 > 0.0 && t1 < n.w)
            {
                p = E + t1 * D;
                n = make_float4(p, t1);
            }
            if (solutions > 1 && t2 < n.w && t2 > 0.0)
            {
                p = E + t2 * D;
                n = make_float4(p, t2);
            }
        }

        return n;
    }



    __device__
    float4 intersectQuad(float3 E, float3 D)
    {
        float t = (-E.z) / D.z;
        float3 p = E + t * D;
        if (t < INF && t > 0.0)
        {
            if (p.x >= -1.0 && p.x <= 1.0 && p.y >= -1.0 && p.y <= 1.0)
                return make_float4(0, 0, (D.z < 0.0 ? 1 : -1), t);
        }

        return make_float4(0, 0, 0, INF);
    }

    // check for intersections with every shape except the excluded one
    __device__
    int intersectWorld(Ray r, Shape *shapes, uint numShapes, float4 &n, Shape &s, int exclude)
    {
        n = make_float4(INF);
        float4 tempN = make_float4(INF);
        int index = -1;


        for (int i = 0; i < MAX_DEVICE_SHAPES; ++i)
        {
            if (i >= numShapes)
                break;

            if (i == exclude)
                continue;

            Shape &shape = shapes[i];

            float3 E = make_float3(((float4 *)shape.inv) * make_float4(r.orig, 1.0));
            float3 D = make_float3(((float4 *)shape.inv) * make_float4(r.dir, 0.0));

            // check bounding box first
//            if (!intersectCubeQuick(E, D, INF))
//                continue;

            switch (shape.type)
            {
//            case CONE:
//                tempN = intersectCone(E, D);
//                if (tempN.w < n.w)
//                {
//                    n = tempN;
//                    index = i;
//                    s = shape;
//                }
//                break;
//            case CUBE:
//                tempN = intersectCube(E, D);
//                if (tempN.w < n.w)
//                {
//                    n = tempN;
//                    index = i;
//                    s = shape;
//                }
//                break;
//            case CYLINDER:
//                tempN = intersectCylinder(E, D);
//                if (tempN.w < n.w)
//                {
//                    n = tempN;
//                    index = i;
//                    s = shape;
//                }
//                break;
            case SPHERE:
                tempN = intersectSphere(E, D);
                if (tempN.w < n.w)
                {
                    n = tempN;
                    index = i;
                    s = shape;
                }
                break;
            case QUAD:
                tempN = intersectQuad(E, D);
                if (tempN.w < n.w)
                {
                    n = tempN;
                    index = i;
                    s = shape;
                }
                break;
            default:
                break;
            }
        }

//        if (index >= 0)
//        {
//            n.xyz = mat3(transpose(s.inv)) * normalize(n.xyz);
//            n.xyz = normalize(n.xyz);
//        }

        return index;
    }
}
