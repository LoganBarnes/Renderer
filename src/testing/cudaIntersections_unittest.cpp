#include <glm/gtc/matrix_transform.hpp>
#include "CudaFunctions.cuh"
#include "tester.cuh"
#include "renderObjects.hpp"
#include "gtest/gtest.h"

const float FLOAT_ERROR2 = 1e-2;
const float FLOAT_ERROR5 = 1e-5;
const float FLOAT_ERROR7 = 1e-7;

const uint DEFAULT_SAMPLES = 100;

namespace
{

class SamplingTest : public ::testing::Test
{
protected:

    SamplingTest()
    {

        cuda_init(0, NULL, false);

        std::srand(std::time(0));
        uint64_t seed = static_cast<uint64_t>(std::rand());

        // allocate resources
        cuda_allocateArray(reinterpret_cast<void**>(&m_randState), DEFAULT_SAMPLES * sizeof(curandState));
        cuda_initCuRand(m_randState, seed, dim3(DEFAULT_SAMPLES, 1));

        cuda_allocateArray(reinterpret_cast<void**>(&m_dResults), DEFAULT_SAMPLES * sizeof(float3));

        cuda_allocateArray(reinterpret_cast<void**>(&m_dShape), sizeof(Shape));
    }

    virtual ~SamplingTest()
    {
        // free resources
        cuda_freeArray(m_dShape);
        cuda_freeArray(m_dResults);
        cuda_freeArray(m_randState);

        cuda_destroy();
    }

    curandState *m_randState;
    float3 *m_dResults; // first half contains points
    Shape *m_dShape;

    void testSamplePoints(glm::mat4 trans, float3 max, float3 min)
    {
        Shape hShape;
        hShape.type = QUAD;
        set_float_mat4(hShape.trans, trans);
        cuda_copyArrayToDevice(m_dShape, &hShape, 0, sizeof(Shape));

        // run test
        cuda_testSamplePoint(m_randState, m_dShape, m_dResults, DEFAULT_SAMPLES, false);

        float3 hResults[DEFAULT_SAMPLES];
        cuda_copyArrayFromDevice(hResults, m_dResults, DEFAULT_SAMPLES * sizeof(float3));

        for (uint i = 0; i < DEFAULT_SAMPLES; ++i)
        {
            float3 &f3 = hResults[i];
            EXPECT_LT(f3.x, max.x);
            EXPECT_LT(f3.y, max.y);
            EXPECT_LT(f3.z, max.z);
            EXPECT_GT(f3.x, min.x);
            EXPECT_GT(f3.y, min.y);
            EXPECT_GT(f3.z, min.z);
        }
    }

    void testSampleNormals(glm::mat4 trans, float3 expected)
    {
        Shape hShape;
        hShape.type = QUAD;
        set_float_mat4(hShape.trans, trans);
        set_float_mat3(hShape.normInv, glm::inverse(glm::transpose(glm::mat3(trans))));
        cuda_copyArrayToDevice(m_dShape, &hShape, 0, sizeof(Shape));

        // run test
        cuda_testSamplePoint(m_randState, m_dShape, m_dResults, DEFAULT_SAMPLES, true);

        float3 hResults[DEFAULT_SAMPLES];
        cuda_copyArrayFromDevice(hResults, m_dResults, DEFAULT_SAMPLES * sizeof(float3));

        for (uint i = 0; i < DEFAULT_SAMPLES; ++i)
        {
            float3 &f3 = hResults[i];
            EXPECT_NEAR(f3.x, expected.x, FLOAT_ERROR7);
            EXPECT_NEAR(f3.y, expected.y, FLOAT_ERROR7);
            EXPECT_NEAR(f3.z, expected.z, FLOAT_ERROR7);
        }
    }

};



/**
 * @brief TEST_F
 */
TEST_F(SamplingTest, SampleQuadWithinBoundaries)
{
    testSamplePoints(glm::mat4(), make_float3(1.f, 1.f, FLOAT_ERROR7), make_float3(-1.f, -1.f, -FLOAT_ERROR7));

    // set transformed test shape 1
    glm::mat4 trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    // FAILS?
    testSamplePoints(trans, make_float3(0.5f, 2.5f + FLOAT_ERROR5, -1.2f), make_float3(-0.5f, 2.5f - FLOAT_ERROR5, -1.8f));

    // set transformed test shape 2
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(45.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    float yLen = std::sin(M_PI * 0.5f) * 0.3f;
    float yMax = 2.5f + yLen;
    float yMin = 2.5f - yLen;

    Shape hShape;
    hShape.type = QUAD;
    set_float_mat4(hShape.trans, trans);
    cuda_copyArrayToDevice(m_dShape, &hShape, 0, sizeof(Shape));

    // run test
    cuda_testSamplePoint(m_randState, m_dShape, m_dResults, DEFAULT_SAMPLES, false);

    float3 hResults[DEFAULT_SAMPLES];
    cuda_copyArrayFromDevice(hResults, m_dResults, DEFAULT_SAMPLES * sizeof(float3));

    for (uint i = 0; i < DEFAULT_SAMPLES; ++i)
    {
        float3 &f3 = hResults[i];
        EXPECT_LT(f3.x, 0.5f);
        EXPECT_LT(f3.y, yMax);
        EXPECT_GT(f3.x, -0.5f);
        EXPECT_GT(f3.y, yMin);
        EXPECT_NEAR(f3.z + 1.5f, f3.y - 2.5f, FLOAT_ERROR5);
    }

}


/**
 * @brief TEST_F
 */
TEST_F(SamplingTest, SampleQuadNormalsCorrect)
{

    // set non-transformed test shape
    glm::mat4 trans = glm::mat4();
    testSampleNormals(trans, make_float3(0.f, 0.f, -1.f));


    // set transformed test shape 1
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    testSampleNormals(trans, make_float3(0.f, -1.f, 0.f));

    // set transformed test shape 2
    trans = glm::mat4();
    trans *= glm::translate(glm::mat4(), glm::vec3(0.f, 2.5f, -1.5f));
    trans *= glm::rotate(glm::mat4(), glm::radians(-45.f), glm::vec3(1.f, 0.f, 0.f));
    trans *= glm::scale(glm::mat4(), glm::vec3(0.5f, 0.3f, 1.f));

    testSampleNormals(trans, normalize(make_float3(0.f, -1.f, -1.f)));
}


///**
// * @brief TEST_F
// */
//TEST_F(SamplingTest, SampleQuadPredictsPI)
//{
//    const uint numSamples = 100000;

//    std::srand(std::time(0));
//    uint64_t seed = static_cast<uint64_t>(std::rand());

//    // allocate resources
//    curandState *randState;
//    cuda_allocateArray(reinterpret_cast<void**>(&randState), numSamples * sizeof(curandState));
//    cuda_initCuRand(randState, seed, dim3(numSamples, 1));

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

//    float pi = static_cast<float>(countIn) * (4.f / static_cast<float>(numSamples));
//    EXPECT_NEAR(M_PI, pi, FLOAT_ERROR2);

//    std::cout << pi << std::endl;


//    // free resources
//    cuda_freeArray(dShape);
//    cuda_freeArray(dResults);
//    cuda_freeArray(randState);

//}



// The fixture for testing class Foo.
class IntersectionTest : public ::testing::Test
{
protected:

    IntersectionTest()
    {
        cuda_init(0, NULL, false);

        // allocate resources
        cuda_allocateArray(reinterpret_cast<void**>(&m_dSurfel), sizeof(SurfaceElement));
        cuda_allocateArray(reinterpret_cast<void**>(&m_dShape), sizeof(Shape));
        cuda_allocateArray(reinterpret_cast<void**>(&m_dRay), sizeof(Ray));
    }

    virtual ~IntersectionTest()
    {
        // free resources
        cuda_freeArray(m_dRay);
        cuda_freeArray(m_dShape);
        cuda_freeArray(m_dSurfel);

        cuda_destroy();
    }

    SurfaceElement *m_dSurfel; // first half contains points
    Shape *m_dShape;
    Ray *m_dRay;

    void testIntersectionNormals(ShapeType type,
                                 glm::mat4 trans,
                                 Ray *hRay,
                                 float3 expected)
    {
        // set shape
        Shape hShape;
        hShape.type = type;
        set_float_mat4(hShape.trans, trans);
        set_float_mat4(hShape.inv, glm::inverse(trans));
        set_float_mat3(hShape.normInv, glm::inverse(glm::transpose(glm::mat3(trans))));
        cuda_copyArrayToDevice(m_dShape, &hShape, 0, sizeof(Shape));

        // set ray
        cuda_copyArrayToDevice(m_dRay, hRay, 0, sizeof(Ray));

        // run test
        cuda_testSphereIntersect(m_dShape, m_dSurfel, m_dRay);

        SurfaceElement hSurfel;
        cuda_copyArrayFromDevice(&hSurfel, m_dSurfel, sizeof(SurfaceElement));

        EXPECT_EQ(777, hSurfel.index);

        float3 &norm = hSurfel.normal;
        EXPECT_NEAR(expected.x, norm.x, FLOAT_ERROR5);
        EXPECT_NEAR(expected.y, norm.y, FLOAT_ERROR5);
        EXPECT_NEAR(expected.z, norm.z, FLOAT_ERROR5);
    }


};


/**
 * @brief TEST_F
 */
TEST_F(IntersectionTest, SphereNormalsNonTransformedCorrect)
{
    std::srand(std::time(0));

    glm::mat4 trans = glm::mat4();
    Ray hRay;
    float3 expected;
    float3 sp; // sphere point

    /*
     * straight on
     */
    hRay.orig = make_float3(0.f, 0.f, -5.f);
    hRay.dir = make_float3(0.f, 0.f, 1.f);
    expected = make_float3(0.f, 0.f, -1.f);
    testIntersectionNormals(SPHERE,
                            trans,
                            &hRay,
                            expected);

    /*
     * straight on translated
     */
    hRay.orig = make_float3(0.5f, 0.5f, -5.f);
    hRay.dir = make_float3(0.f, 0.f, 1.f);
    expected = normalize(make_float3(0.5f, 0.5f, -std::sqrtf(1.f - 0.5f)));
    testIntersectionNormals(SPHERE,
                            trans,
                            &hRay,
                            expected);

    /*
     * angled
     */
    hRay.orig = make_float3(1.f);
    hRay.dir = make_float3(-1.f);
    expected = normalize(make_float3(1.f));
    testIntersectionNormals(SPHERE,
                            trans,
                            &hRay,
                            expected);

    /*
     * angled translated
     */
    // random positive sphere point
    sp = make_float3(std::rand(), std::rand(), std::rand());
    sp *= (1.f / static_cast<float>(RAND_MAX));
    sp = normalize(sp);

    hRay.orig = sp + make_float3(1.f);
    hRay.dir = make_float3(-1.f);
    expected = sp;
    testIntersectionNormals(SPHERE,
                            trans,
                            &hRay,
                            expected);

    // random negative sphere point
    sp = make_float3(std::rand(), std::rand(), std::rand());
    sp *= (-1.f / static_cast<float>(RAND_MAX));
    sp = normalize(sp);

    hRay.orig = sp + make_float3(-1.f);
    hRay.dir = make_float3(1.f);
    expected = sp;
    testIntersectionNormals(SPHERE,
                            trans,
                            &hRay,
                            expected);

}


/**
 * @brief TEST_F
 */
TEST_F(IntersectionTest, SphereNormalsTransformedCorrect)
{
//    std::srand(std::time(0));

    glm::mat4 trans = glm::mat4();
    Ray hRay;
    float3 expected;
//    float3 sp; // sphere point

    /*
     * straight on
     */
    hRay.orig = make_float3(0.f, 0.f, -5.f);
    hRay.dir = make_float3(0.f, 0.f, 1.f);
    expected = make_float3(0.f, 0.f, -1.f);
    testIntersectionNormals(SPHERE,
                            trans,
                            &hRay,
                            expected);

}


} // namespace
