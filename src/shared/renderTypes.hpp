#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

const unsigned int MAX_DEVICE_SHAPES = 10;
const unsigned int MAX_DEVICE_LUMINAIRES = 5;

enum ShapeType
{
    AABB, CONE, CUBE, CYLINDER, QUAD, SPHERE, TRIANGLE, NUM_SHAPE_TYPES
};

enum LuminaireType
{
    POINT, DIRECTIONAL, AREA, SPOT, NUM_LUMINAIRE_TYPES
};

#endif // RENDER_TYPES_H