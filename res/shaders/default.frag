#version 410 core

uniform sampler2D uTexture;

in vec2 vTexCoords;

out vec4 fragColor;

void main(void)
{
    vec3 color = texture(uTexture, vTexCoords).rgb;
    fragColor = vec4(color, 1.0);
}
