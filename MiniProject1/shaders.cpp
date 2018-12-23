/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(
//layout(location = 0) in vec3 aPos;
//layout(location = 1) in vec3 aNormal;

//out vec3 Normal;
varying vec3 Position;
varying vec3 camera;

uniform mat4 model;
//uniform mat4 projection;
                               uniform float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);
//	Normal = mat3(transpose(inverse(model))) * aNormal;
	mat3 rotMat = mat3(gl_ModelViewMatrix);
	vec3 d = vec3(gl_ModelViewMatrix[3]);

	camera = -d * rotMat;

	Position = vec3(model * vec4(gl_Vertex.xyz, 1.0));
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
//	Position = vec3(vec4(gl_Vertex.xyz, 1.0));

    //gl_FrontColor = gl_Color;
	//gl_FrontColor = vec3(0.0, 0.0, 0.0);
}
                           );

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(
//out vec4 FragColor;

//in vec3 Normal;
varying vec3 Position;
varying vec3 camera;

uniform samplerCube skybox;

                                    void main()
{
   // const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    // calculate lighting
    //float diffuse = max(0.0, dot(lightDir, N));
	vec3 I = normalize(Position - camera);

	//vec2 texCoord = gl_TexCoord[0].st;
	//vec3 color = texture2D(tex, texCoord).rgb;
	//if (length(color) < 0.01f) discard;
	//gl_FragColor = vec4(color, 1.0f);

	vec3 reflectRay = reflect(I, normalize(N));
	vec3 refractRay = refract(I, normalize(N), 0.65f);
	gl_FragColor = vec4(texture(skybox, reflectRay).rgb, 1.0);
}
                                );

