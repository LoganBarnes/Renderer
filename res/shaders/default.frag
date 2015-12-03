#version 410 core

uniform sampler2D uTexture;

in vec2 vTexCoords;

out vec4 fragColor;

void main(void)
{
    vec4 color = texture(uTexture, vTexCoords);
    fragColor = color;
}
