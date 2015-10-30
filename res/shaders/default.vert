#version 410 core

in vec2 aUV;

out vec2 vTexCoords;

void main(void)
{
    vTexCoords = aUV * vec2(0.5) + vec2(0.5);
    gl_Position = vec4(aUV, -1.0, 1.0);
}
