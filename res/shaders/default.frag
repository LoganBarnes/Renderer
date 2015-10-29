#version 410 core

in vec2 vUV;

out vec4 fragColor;

void main(void)
{
    fragColor = vec4(vUV, 0.0, 1.0);
}
