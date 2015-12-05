#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

const unsigned int MAX_SHAPES = 15;
const unsigned int MAX_AREA_LIGHTS = 5;

enum ShapeType
{
    AABB,
    CONE,
    CUBE,
    CYLINDER,
    QUAD,
    SPHERE,
    TRIANGLE,
    NUM_SHAPE_TYPES
};

#endif // RENDER_TYPES_H
