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
//    bvec3 nans = isnan(color);
//    if (nans.x || nans.y || nans.z)
//        color = vec3(0.0, 0.0, 1.0);

//    bvec3 infs = isinf(color);
//    if (infs.x || infs.y || infs.z)
//        color = vec3(0.0, 1.0, 1.0);

//    if (color.x > 1.0 || color.y > 1.0 || color.z > 1.0)
//        color = vec3(0.0);

    fragColor = vec4(color, 1.0);
}
