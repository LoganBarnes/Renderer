#include <glm/gtc/matrix_transform.hpp>
#include "CudaFunctions.cuh"
#include "tester.cuh"
#include "renderObjects.hpp"
#include "gtest/gtest.h"

const float FLOAT_ERROR2 = 1e-2;
const float FLOAT_ERROR5 = 1e-5;
const float FLOAT_ERROR7 = 1e-7;

namespace
{

// The fixture for testing class Foo.
class IntersectionTest : public ::testing::Test
{
protected:

    IntersectionTest()
    {
        cuda_init(0, NULL, false);
    }

    virtual ~IntersectionTest()
    {
        cuda_destroy();
    }

};


/**
 * @brief TEST_F
 */
TEST_F(IntersectionTest, SampleQuadWithinBoundaries)
{
    const uint numSamples = 100;

    // allocate resources
    curandState *randState;
    cuda_allocateArray(reinterpret_cast<void**>(&randState), numSamples * sizeof(curandState));
    cuda_initCuRand(randState, 1337, dim3(numSamples, 1));

    float3 *dResults; // first half contains points
    cuda_allocateArray(reinterpret_cast<void**>(&dResults), numSamples * sizeof(float3));

    Shape *dShape;
    cuda_allocateArray(reinterpret_cast<void**>(&dShape), sizeof(Shape));


    // set non-transformed test shape
    Shape hShape;
    hShape.type = QUAD;
    set_float_mat4(hShape.trans, glm::mat4());
    cuda_copyArrayToDevice(dShape, &hShape, 0, sizeof(Shape));

    // run test
    cuda_testSamplePoint(randState, dShape, dResults, numSamples, false);

    float3 hResults[numSamples];
    cuda_copyArrayFromDevice(hResults, dResults, numSamples * sizeof(float3));

    for (uint i = 0; i < numSamples; ++i)
    {
        float3 &f3 = hResults[i];
        EXPECT_LT(f3.x, 1.f);
        EXPECT_LT(f3.y, 1.f);
        EXPECT_GT(f3.x, -1.f);
        EXPECT_GT(f3.y, -1.f);
        EXPECT_EQ(f3.z, 0.f);
    }


    // set transformed test shape 1
    glm::mat4 trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    set_float_mat4(hShape.trans, trans);
    cuda_copyArrayToDevice(dShape, &hShape, 0, sizeof(Shape));

    // run test
    cuda_testSamplePoint(randState, dShape, dResults, numSamples, false);

    cuda_copyArrayFromDevice(hResults, dResults, numSamples * sizeof(float3));

    for (uint i = 0; i < numSamples; ++i)
    {
        float3 &f3 = hResults[i];
        EXPECT_LT(f3.x, 0.5f);
        EXPECT_LT(f3.z, -1.2f);
        EXPECT_GT(f3.x, -0.5f);
        EXPECT_GT(f3.z, -1.8f);
        EXPECT_FLOAT_EQ(f3.y, 2.5f);
    }


    // set transformed test shape 2
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(45.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    set_float_mat4(hShape.trans, trans);
    cuda_copyArrayToDevice(dShape, &hShape, 0, sizeof(Shape));

    // run test
    cuda_testSamplePoint(randState, dShape, dResults, numSamples, false);

    cuda_copyArrayFromDevice(hResults, dResults, numSamples * sizeof(float3));

    float yLen = std::sin(M_PI * 0.5f) * 0.3f;
    float yMax = 2.5f + yLen;
    float yMin = 2.5f - yLen;
    for (uint i = 0; i < numSamples; ++i)
    {
        float3 &f3 = hResults[i];
        EXPECT_LT(f3.x, 0.5f);
        EXPECT_LT(f3.y, yMax);
        EXPECT_GT(f3.x, -0.5f);
        EXPECT_GT(f3.y, yMin);
        EXPECT_NEAR(f3.z + 1.5f, f3.y - 2.5f, FLOAT_ERROR5);
    }


    // free resources
    cuda_freeArray(dShape);
    cuda_freeArray(dResults);
    cuda_freeArray(randState);

}


/**
 * @brief TEST_F
 */
TEST_F(IntersectionTest, SampleQuadNormalsCorrect)
{
    const uint numSamples = 100;

    // allocate resources
    curandState *randState;
    cuda_allocateArray(reinterpret_cast<void**>(&randState), numSamples * sizeof(curandState));
    cuda_initCuRand(randState, 1337, dim3(numSamples, 1));

    float3 *dResults; // first half contains points
    cuda_allocateArray(reinterpret_cast<void**>(&dResults), numSamples * sizeof(float3));

    Shape *dShape;
    cuda_allocateArray(reinterpret_cast<void**>(&dShape), sizeof(Shape));


    // set non-transformed test shape
    Shape hShape;
    hShape.type = QUAD;
    set_float_mat4(hShape.trans, glm::mat4());
    cuda_copyArrayToDevice(dShape, &hShape, 0, sizeof(Shape));

    // run test
    cuda_testSamplePoint(randState, dShape, dResults, numSamples, true);

    float3 hResults[numSamples];
    cuda_copyArrayFromDevice(hResults, dResults, numSamples * sizeof(float3));

    for (uint i = 0; i < numSamples; ++i)
    {
        float3 &f3 = hResults[i];
        EXPECT_NEAR(f3.x, 0.f, FLOAT_ERROR7);
        EXPECT_NEAR(f3.y, 0.f, FLOAT_ERROR7);
        EXPECT_NEAR(f3.z, 1.f, FLOAT_ERROR7);
    }


    // set transformed test shape 1
    glm::mat4 trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    set_float_mat4(hShape.trans, trans);
    cuda_copyArrayToDevice(dShape, &hShape, 0, sizeof(Shape));

    // run test
    cuda_testSamplePoint(randState, dShape, dResults, numSamples, true);

    cuda_copyArrayFromDevice(hResults, dResults, numSamples * sizeof(float3));

    for (uint i = 0; i < numSamples; ++i)
    {
        float3 &f3 = hResults[i];
        EXPECT_NEAR(f3.x, 0.f, FLOAT_ERROR7);
        EXPECT_NEAR(f3.y, -1.f, FLOAT_ERROR7);
        EXPECT_NEAR(f3.z, 0.f, FLOAT_ERROR7);
    }


    // set transformed test shape 2
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(45.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    set_float_mat4(hShape.trans, trans);
    cuda_copyArrayToDevice(dShape, &hShape, 0, sizeof(Shape));

    // run test
    cuda_testSamplePoint(randState, dShape, dResults, numSamples, true);

    cuda_copyArrayFromDevice(hResults, dResults, numSamples * sizeof(float3));

    float invSqrt2 = 1.f / std::sqrtf(2.f);
    for (uint i = 0; i < numSamples; ++i)
    {
        float3 &f3 = hResults[i];
        EXPECT_NEAR(f3.x, 0.f, FLOAT_ERROR7);
        EXPECT_NEAR(f3.y, -invSqrt2, FLOAT_ERROR7);
        EXPECT_NEAR(f3.z, invSqrt2, FLOAT_ERROR7);
    }


    // free resources
    cuda_freeArray(dShape);
    cuda_freeArray(dResults);
    cuda_freeArray(randState);

}


///**
// * @brief TEST_F
// */
//TEST_F(IntersectionTest, SampleQuadPredictsPI)
//{
//    const uint numSamples = 40000;

//    // allocate resources
//    curandState *randState;
//    cuda_allocateArray(reinterpret_cast<void**>(&randState), numSamples * sizeof(curandState));
//    cuda_initCuRand(randState, 1337, dim3(numSamples, 1));

//    float3 *dResults; // first half contains points
//    cuda_allocateArray(reinterpret_cast<void**>(&dResults), numSamples * sizeof(float3));

//    Shape *dShape;
//    cuda_allocateArray(reinterpret_cast<void**>(&dShape), sizeof(Shape));


//    // set non-transformed test shape
//    Shape hShape;
//    hShape.type = QUAD;
//    set_float_mat4(hShape.trans, glm::mat4());
//    cuda_copyArrayToDevice(dShape, &hShape, 0, sizeof(Shape));

//    // run test
//    cuda_testSamplePoint(randState, dShape, dResults, numSamples, false);

//    float3 hResults[numSamples];
//    cuda_copyArrayFromDevice(hResults, dResults, numSamples * sizeof(float3));

//    uint countIn = 0;
//    for (uint i = 0; i < numSamples; ++i)
//    {
//        float3 &f3 = hResults[i];
//        if (dot(f3, f3) < 1.f)
//            ++countIn;
//    }

//    float pi = static_cast<float>(countIn) / 10000.f;
//    EXPECT_NEAR(M_PI, pi, FLOAT_ERROR2);


//    // free resources
//    cuda_freeArray(dShape);
//    cuda_freeArray(dResults);
//    cuda_freeArray(randState);

//}


} // namespace
