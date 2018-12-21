#define STRINGIFY(A) #A

const char *vertexObj = STRINGIFY(
#version 130
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
out vec3 Normal;
out vec3 Position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
	Normal = mat3(transpose(inverse(model))) * aNormal;
	Position = vec3(model * vec4(aPos, 1.0));
	gl_Position = projection * view * model * vec4(aPos, 1.0);
}

);

const char *fragObj = STRINGIFY(
#version 130

out vec4 FragColor;
in vec3 Normal;
in vec3 Position;
uniform vec3 cameraPos;
uniform samplerCube skybox;
void main()
{
	vec3 I = normalize(Position - cameraPos);
	vec3 R = reflect(I, normalize(Normal));
	vec3 reflectRay = reflect(I, normalize(Normal));
	vec3 refractRay = refract(I, normalize(Normal), 0.6f);
	FragColor = vec4(texture(skybox, reflectRay).rgb, 1.0);
}

);

const char *vertexSkybox = STRINGIFY(
	#version 410\n


layout(location = 0) in vec3 aPos;
out vec3 TexCoords;
uniform mat4 projection;
uniform mat4 view;
void main()
{
	TexCoords = aPos;
	vec4 pos = projection * view * vec4(aPos, 1.0);
	gl_Position = pos.xyww;
}

);

const char *fragSkybox = STRINGIFY(
	#version 410\n


out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{
	FragColor = texture(skybox, TexCoords);
}

);