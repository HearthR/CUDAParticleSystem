#include <math.h>
#include <assert.h>
#include <stdio.h>
// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include "helper_gl.h"
#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer(): m_pos(0), m_numParticles(0), m_pointSize(1.0f), m_particleRadius(0.125f * 0.5f), m_program(0), m_vbo(0), m_colorVBO(0){
    _initGL();
}
ParticleRenderer::~ParticleRenderer(){
    m_pos = 0;
}
void ParticleRenderer::setPositions(float *pos, int numParticles){
    m_pos = pos;
    m_numParticles = numParticles;
}
void ParticleRenderer::setVertexBuffer(unsigned int vbo, int numParticles){
    m_vbo = vbo;
    m_numParticles = numParticles;
}
void ParticleRenderer::_drawPoints(){
    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
		glActiveTexture(GL_TEXTURE0);
		//glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapIdcu);
		glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
		glBindTexture(GL_TEXTURE_2D, smokeId);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        /*if (m_colorVBO){
            glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }*/
        glDrawArrays(GL_POINTS, 0, m_numParticles);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        //glDisableClientState(GL_COLOR_ARRAY);
    }
}
void ParticleRenderer::display(float *cameraPos, GLuint cubeMap, GLuint smokeTex, DisplayMode mode){
	float* model = (float*)malloc(16 * sizeof(float));
	float* view = (float*)malloc(16 * sizeof(float));
	float* projection = (float*)malloc(16 * sizeof(float));
	cubeMapIdcu = cubeMap;

	glGetFloatv(GL_MODELVIEW_MATRIX, model);
	glGetFloatv(GL_PROJECTION_MATRIX, projection);
    switch (mode)
    {
        case PARTICLE_POINTS:
            glColor3f(1, 1, 1);
            glPointSize(m_pointSize);
            _drawPoints();
            break;

        default:
        case PARTICLE_SPHERES:
            glEnable(GL_POINT_SPRITE_ARB);
            glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);

            glUseProgram(m_program);
            glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
            glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);
			glUniformMatrix4fv(glGetUniformLocation(m_program, "model"), 1, GL_FALSE, &model[0]);
//			glUniformMatrix4fv(glGetUniformLocation(m_program, "view"), 1, GL_FALSE, &view[0]);
//			glUniformMatrix4fv(glGetUniformLocation(m_program, "projection"), 1, GL_FALSE, &projection[0]);
			glUniform3f(glGetUniformLocation(m_program, "cameraPos"), (GLfloat)(cameraPos[0]), (GLfloat)(cameraPos[1]), (GLfloat)(cameraPos[2]));
            glColor3f(1, 1, 1);
            _drawPoints();

            glUseProgram(0);
            glDisable(GL_POINT_SPRITE_ARB);
            break;
    }

	free(model);
	free(view);
	free(projection);
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    m_program = _compileProgram(vertexShader, spherePixelShader);
}
