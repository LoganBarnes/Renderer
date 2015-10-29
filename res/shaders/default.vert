#version 410 core

in vec2 aUV;

out vec2 vUV;

void main(void)
{
    vUV = aUV;
    gl_Position = vec4(aUV, -1.0, 1.0);
}
