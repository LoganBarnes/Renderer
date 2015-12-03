#include <cstdlib>
#include <glm/gtc/type_ptr.hpp>
#include "renderObjects.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

const float FLOAT_ERROR = 0.0001f;
const float FLOAT_ERROR_PRECISE = 1e-10;

namespace
{

// The fixture for testing class Foo.
class MathTest : public ::testing::Test
{};

// used for comparing float equality up to the given epsilon value
using ::testing::get;
MATCHER_P(FloatEq, eps, "") {
    return std::fabs(get<0>(arg) - get<1>(arg)) < eps;
}


/*
 * Tests stuff
 */
TEST_F(MathTest, MakeMatMimicsGLMMat)
{
    using ::testing::ElementsAreArray;
    using ::testing::Pointwise;

    std::srand(std::time(0));

    // integers
    glm::mat4 glmMat4 = glm::mat4(std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand());
    glm::mat3 glmMat3 = glm::mat3(glmMat4);
    float4 myMat4[4];
    set_float_mat4(myMat4, glmMat4);
    float3 myMat3[3];
    set_float_mat3(myMat3, glmMat3);

    float *glm4Ptr = glm::value_ptr(glmMat4);
    float *my4Ptr = reinterpret_cast<float*>(myMat4);
    float *glm3Ptr = glm::value_ptr(glmMat3);
    float *my3Ptr = reinterpret_cast<float*>(myMat3);

    // convert to vectors for comparison
    std::vector<float> glmInt4Vector(glm4Ptr, glm4Ptr + 16);
    std::vector<float> myInt4Vector(my4Ptr, my4Ptr + 16);
    EXPECT_THAT(myInt4Vector, Pointwise(FloatEq(FLOAT_ERROR_PRECISE), glmInt4Vector));
    // convert to vectors for comparison
    std::vector<float> glmInt3Vector(glm3Ptr, glm3Ptr + 9);
    std::vector<float> myInt3Vector(my3Ptr, my3Ptr + 9);
    EXPECT_THAT(myInt3Vector, Pointwise(FloatEq(FLOAT_ERROR_PRECISE), glmInt3Vector));



    // floats from 0 - 1 inclusive
    glmMat4 *= 1.f / static_cast<float>(RAND_MAX);
    set_float_mat4(myMat4, glmMat4);

    // convert to vectors for comparison
    std::vector<float> glmFloat4Vector(glm4Ptr, glm4Ptr + 16);
    std::vector<float> myFloat4Vector(my4Ptr, my4Ptr + 16);
    EXPECT_THAT(glmFloat4Vector, Pointwise(FloatEq(FLOAT_ERROR_PRECISE), myFloat4Vector));


    // convert to mat3
    glmMat3 = glm::mat3(glmMat4);
    make_float_mat3(myMat3, myMat4);

    glm3Ptr = glm::value_ptr(glmMat3);
    my3Ptr = reinterpret_cast<float*>(myMat3);

    // convert to vectors for comparison
    std::vector<float> glmFloat3Vector(glm3Ptr, glm3Ptr + 9);
    std::vector<float> myFloat3Vector(my3Ptr, my3Ptr + 9);
    EXPECT_THAT(glmFloat3Vector, Pointwise(FloatEq(FLOAT_ERROR_PRECISE), myFloat3Vector));

}


/*
 * Tests stuff
 */
TEST_F(MathTest, MatrixVectorMult4EqualsGLMMatrixVectorMult4)
{
    using ::testing::ElementsAreArray;
    using ::testing::Pointwise;

    // set vector arrays to compare later
    const int numTestVecs = 1000;
    glm::vec4 glmResults[numTestVecs];
    float4 myResults[numTestVecs];

    // matrices to multiply
    glm::mat4 glmMat4 = glm::mat4(std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand());
    glmMat4 *= 1.f / static_cast<float>(RAND_MAX);
    float4 myMat4[4];
    set_float_mat4(myMat4, glmMat4);


    // multiply random vectors by both matrices and store results
    float invMax = 1.f / RAND_MAX;
    float x, y, z, w;
    for (int i = 0; i < numTestVecs; ++i)
    {
        x = std::rand() * invMax;
        y = std::rand() * invMax;
        z = std::rand() * invMax;
        w = std::rand() * invMax;

        glmResults[i] = glmMat4 * glm::vec4(x, y, z, w);
        myResults[i] = myMat4 * make_float4(x, y, z, w);
    }

    // get pointers to arrays
    float *glmPtr = reinterpret_cast<float*>(glmResults);
    float *myPtr = reinterpret_cast<float*>(myResults);

    // convert to vectors for comparison
    std::vector<float> glmFloatVector(glmPtr, glmPtr + 4 * numTestVecs);
    std::vector<float> myFloatVector(myPtr, myPtr + 4 * numTestVecs);
    EXPECT_THAT(glmFloatVector, Pointwise(FloatEq(FLOAT_ERROR), myFloatVector));
}


/*
 * Tests stuff
 */
TEST_F(MathTest, MatrixVectorMult3EqualsGLMMatrixVectorMult3)
{
    using ::testing::ElementsAreArray;
    using ::testing::Pointwise;

    // set vector arrays to compare later
    const int numTestVecs = 1000;
    glm::vec3 glmResults[numTestVecs];
    float3 myResults[numTestVecs];

    // matrices to multiply
    glm::mat4 glmMat4 = glm::mat4(std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand(),
                                  std::rand(), std::rand(), std::rand(), std::rand());
    glmMat4 *= 1.f / static_cast<float>(RAND_MAX);
    float4 myMat4[4];
    float3 myMat3[3];
    set_float_mat4(myMat4, glmMat4);
    make_float_mat3(myMat3, myMat4);
    glm::mat3 glmMat3 = glm::mat3(glmMat4);


    // multiply random vectors by both matrices and store results
    float invMax = 1.f / RAND_MAX;
    float x, y, z;
    for (int i = 0; i < numTestVecs; ++i)
    {
        x = std::rand() * invMax;
        y = std::rand() * invMax;
        z = std::rand() * invMax;

        glmResults[i] = glmMat3 * glm::vec3(x, y, z);
        myResults[i] = myMat3 * make_float3(x, y, z);
    }

    // get pointers to arrays
    float *glmPtr = reinterpret_cast<float*>(glmResults);
    float *myPtr = reinterpret_cast<float*>(myResults);

    // convert to vectors for comparison
    std::vector<float> glmFloatVector(glmPtr, glmPtr + 3 * numTestVecs);
    std::vector<float> myFloatVector(myPtr, myPtr + 3 * numTestVecs);
    EXPECT_THAT(glmFloatVector, Pointwise(FloatEq(FLOAT_ERROR), myFloatVector));
}

} // namespace
