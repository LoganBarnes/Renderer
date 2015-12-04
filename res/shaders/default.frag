#version 410 core

uniform sampler2D uTexture;
uniform sampler2D uBlendTex;
uniform int uIteration;

in vec2 vTexCoords;

out vec4 fragColor;

void main(void)
{
    vec3 color = texture(uTexture, vTexCoords).rgb;
    if (uIteration > 0)
    {
        vec3 blend = texture(uBlendTex, vTexCoords).rgb;
        float alpha = 1.0 / float(uIteration);
        color = color * alpha + blend * (1.0 - alpha);
    }

    fragColor = vec4(color, 1.0);
}
