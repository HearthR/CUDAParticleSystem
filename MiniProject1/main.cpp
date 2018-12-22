// OpenGL Graphics includes
#include "helper_gl.h"
#include <GL/freeglut.h>
// CUDA
#include <cuda_runtime.h>
#include "helper_functions.h"
#include "helper_cuda.h"    // includes cuda.h and cuda_runtime_api.h
// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"
#include "Bitmap.h"
#include "additional_shaders.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f
#define GRID_SIZE       64
#define NUM_PARTICLES   450000

//Global Variables
const uint width = 640, height = 480;
//Camera
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;
int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 2000;
enum { M_VIEW = 0, M_MOVE };
uint numParticles = 0;
uint3 gridSize;
int numIterations = 0; // run until exit
float timestep = 0.5f;
float damping = 1.0f;
float gravity = 0.0003f;
int iterations = 1;
int ballr = 10;
float collideSpring = 0.5f;
float collideDamping = 0.02f;
float collideShear = 0.1f;
float collideAttraction = 0.0f;
ParticleSystem *psystem = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;
ParticleRenderer *renderer = 0;
float modelView[16];
ParamListGL *params;
// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
const char *sSDKsample = "CUDA Particles Simulation";

GLuint cubeMapId;
GLuint skyVAO;
GLuint skyVBO;
GLuint skyShaderProgram;

float3 mpath = {0.8f, 0.8f, 0.8f};
int toggle_dt = 1;
float rotate_path = 0.045;

GLuint smokeId;

void SmokeTexture()
{
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &smokeId);
	glBindTexture(GL_TEXTURE_2D, smokeId);

	const char path[] = "textures/smoke.jpg";
	int n = sizeof(path) / sizeof(*path);
	char* filePath = (char*)malloc(n * sizeof(char));
	strcpy(filePath, path);

	Bitmap bmp = Bitmap::bitmapFromFile(filePath);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.width(), bmp.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.pixelBuffer());
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	//glUniform1i(glGetUniformLocation(renderer->getProgram(), "tex"), 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void CubeMapTexture()
{
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &cubeMapId);

	glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapId);

	Bitmap bmp = Bitmap::bitmapFromFile("textures/right.jpg");
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, GL_RGB, bmp.width(), bmp.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.pixelBuffer());

	bmp = Bitmap::bitmapFromFile("textures/left.jpg");
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, GL_RGB, bmp.width(), bmp.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.pixelBuffer());

	bmp = Bitmap::bitmapFromFile("textures/top.jpg");
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, GL_RGB, bmp.width(), bmp.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.pixelBuffer());

	bmp = Bitmap::bitmapFromFile("textures/bottom.jpg");
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, GL_RGB, bmp.width(), bmp.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.pixelBuffer());

	bmp = Bitmap::bitmapFromFile("textures/front.jpg");
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, GL_RGB, bmp.width(), bmp.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.pixelBuffer());

	bmp = Bitmap::bitmapFromFile("textures/back.jpg");
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, GL_RGB, bmp.width(), bmp.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.pixelBuffer());

	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}
void SkyBoxDataTransfer()
{

	float skyboxVertices[] = {
		// positions          
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,
		1.0f,  1.0f, -1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		1.0f, -1.0f,  1.0f
	};

	glGenVertexArrays(1, &skyVAO);
	glGenBuffers(1, &skyVBO);
	glBindVertexArray(skyVAO);
	glBindBuffer(GL_ARRAY_BUFFER, skyVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
}

//Functions
extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);
void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL){
    psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL);
    psystem->reset(ParticleSystem::CONFIG_GRID);

    if (bUseOpenGL)
    {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    sdkCreateTimer(&timer);
}
void cleanup(){
    sdkDeleteTimer(&timer);

    if (psystem)
    {
        delete psystem;
    }
    return;
}
void createProgram(){
	GLuint skyVert = glCreateShader(GL_VERTEX_SHADER);
	GLuint skyFrag = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(skyVert, 1, &vertexSkybox, 0);
	glShaderSource(skyFrag, 1, &fragSkybox, 0);

	glCompileShader(skyVert);
	glCompileShader(skyFrag);

	skyShaderProgram = glCreateProgram();

	glAttachShader(skyShaderProgram, skyVert);
	glAttachShader(skyShaderProgram, skyFrag);

	glLinkProgram(skyShaderProgram);

	GLint success = 0;
	glGetProgramiv(skyShaderProgram, GL_LINK_STATUS, &success);

	if (!success)
	{
		char temp[256];
		glGetProgramInfoLog(skyShaderProgram, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(skyShaderProgram);
		skyShaderProgram = 0;
	}
}
void initGL(int *argc, char **argv){
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
	glutCreateWindow("CUDA Particles");

    if (!isGLVersionSupported(2,0) ||
        !areGLExtensionsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }


	createProgram();

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}
void computeFPS(){
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "CUDA Particles (%d particles): %3.1f fps", numParticles, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}
void display(){
	float* model = (float*)malloc(16 * sizeof(float));
	float* view = (float*)malloc(16 * sizeof(float));
	float* projection = (float*)malloc(16 * sizeof(float));
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);

	view[0] = 1.0;    view[1] = 0.0;    view[2] = 0.0;      view[3] = 0.0;
	view[4] = 0.0;    view[5] = 1.0;    view[6] = 0.0;      view[7] = 0.0;
	view[8] = 0.0;    view[9] = 0.0;    view[10] = 1.0;     view[11] = 0.0;
	view[12] = 0.0f;   view[13] = 0.0;   view[14] = -10.0;     view[15] = 1.0;

    sdkStartTimer(&timer);

    // update the simulation
    if (!bPause)
    {
        psystem->setIterations(iterations);
        psystem->setDamping(damping);
        psystem->setGravity(-gravity);
        psystem->setCollideSpring(collideSpring);
        psystem->setCollideDamping(collideDamping);
        psystem->setCollideShear(collideShear);
        psystem->setCollideAttraction(collideAttraction);

		//rotate_path += 0.01;
		float radian = rotate_path / 3.1415f;

		float3 mpathp;
		float mag;
		mpathp.x = mpath.x;
		mpathp.y = mpath.y;
		mpathp.z = mpath.z;

		mpath.x = mpathp.x * cosf(radian) + mpathp.z * sinf(radian);
		mpath.z = mpathp.z * cosf(radian) - mpathp.x * sinf(radian);

		mag = mpath.x * mpath.x + mpath.y * mpath.y + mpath.z * mpath.z;


/*		if (toggle_dt == 1)
		{
			if (mpath > 1.5f) toggle_dt = 0;
			mpath += 0.015f;
		}
		else
		{
			if (mpath < -1.5f) toggle_dt = 1;
			mpath -= 0.015f;
		}*/
        psystem->update(mpath);

        if (renderer)
        {
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        }
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glGetFloatv(GL_PROJECTION_MATRIX, projection);

    // cube
 //   glColor3f(1.0, 1.0, 1.0);
//    glutWireCube(2.0);

/*	glPushMatrix();
	glPointSize(6);
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_POINTS);
	glVertex3f(mpath.x, mpath.y, mpath.z);
	glEnd();
	glPopMatrix();*/

    // collider
    /*glPushMatrix();
    float3 p = psystem->getColliderPos();
    glTranslatef(p.x, p.y, p.z);
    glColor3f(1.0, 0.0, 0.0);
    glutSolidSphere(psystem->getColliderRadius(), 20, 10);
    glPopMatrix();*/

    if (renderer && displayEnabled)
    {
        renderer->display(camera_trans_lag, cubeMapId, smokeId, displayMode);
    }
	
	//Drawing Skybox
	view[14] = 0.0;
	glDepthFunc(GL_LEQUAL);
	glUseProgram(skyShaderProgram);
	glUniformMatrix4fv(glGetUniformLocation(skyShaderProgram, "view"), 1, GL_FALSE, &view[0]);
	glUniformMatrix4fv(glGetUniformLocation(skyShaderProgram, "projection"), 1, GL_FALSE, &projection[0]);
	glBindVertexArray(skyVAO);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapId);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glDepthFunc(GL_LESS);
	glUseProgram(0);

    if (displaySliders)
    {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    sdkStopTimer(&timer);

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
	free(model);
	free(view);
	free(projection);
}
inline float frand(){
    return rand() / (float) RAND_MAX;
}
void addSphere(){
    // inject a sphere of particles
    float pr = psystem->getParticleRadius();
    float tr = pr+(pr*2.0f)*ballr;
    float pos[4], vel[4];
    pos[0] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[1] = 1.0f - tr;
    pos[2] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[3] = 0.0f;
    vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
    psystem->addSphere(0, pos, vel, ballr, pr*2.0f);
}
void reshape(int w, int h){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer)
    {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}
void mouse(int button, int state, int x, int y){
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    if (displaySliders)
    {
        if (params->Mouse(x, y, button, state))
        {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}
void xform(float *v, float *r, GLfloat *m){
    r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
    r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
    r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}
void ixform(float *v, float *r, GLfloat *m){
    r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
    r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
    r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}
void ixformPoint(float *v, float *r, GLfloat *m){
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}
void motion(int x, int y){
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (displaySliders)
    {
        if (params->Motion(x, y))
        {
            ox = x;
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

    switch (mode)
    {
        case M_VIEW:
            if (buttonState == 3)
            {
                // left+middle = zoom
                camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
            }
            else if (buttonState & 2)
            {
                // middle = translate
                camera_trans[0] += dx / 100.0f;
                camera_trans[1] -= dy / 100.0f;
            }
            else if (buttonState & 1)
            {
                // left = rotate
                camera_rot[0] += dy / 5.0f;
                camera_rot[1] += dx / 5.0f;
            }

            break;

        case M_MOVE:
            {
                float translateSpeed = 0.003f;
                float3 p = psystem->getColliderPos();

                if (buttonState==1)
                {
                    float v[3], r[3];
                    v[0] = dx*translateSpeed;
                    v[1] = -dy*translateSpeed;
                    v[2] = 0.0f;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }
                else if (buttonState==2)
                {
                    float v[3], r[3];
                    v[0] = 0.0f;
                    v[1] = 0.0f;
                    v[2] = dy*translateSpeed;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }

                psystem->setColliderPos(p);
            }
            break;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    glutPostRedisplay();
}
void key(unsigned char key, int /*x*/, int /*y*/){
    switch (key)
    {
        case ' ':
            bPause = !bPause;
            break;

        case 13:
            psystem->update(mpath);

            if (renderer)
            {
                renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
            }

            break;

        case '\033':
        case 'q':
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        case 'v':
            mode = M_VIEW;
            break;

        case 'm':
            mode = M_MOVE;
            break;

        case 'p':
            displayMode = (ParticleRenderer::DisplayMode)
                          ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
            break;

        case 'd':
            psystem->dumpGrid();
            break;

        case 'u':
            psystem->dumpParticles(0, numParticles-1);
            break;

        case 'r':
            displayEnabled = !displayEnabled;
            break;

        case '1':
            psystem->reset(ParticleSystem::CONFIG_GRID);
            break;

        case '2':
            psystem->reset(ParticleSystem::CONFIG_RANDOM);
            break;

        case '3':
            addSphere();
            break;

        case '4':
            {
                // shoot ball from camera
                float pr = psystem->getParticleRadius();
                float vel[4], velw[4], pos[4], posw[4];
                vel[0] = 0.0f;
                vel[1] = 0.0f;
                vel[2] = -0.05f;
                vel[3] = 0.0f;
                ixform(vel, velw, modelView);

                pos[0] = 0.0f;
                pos[1] = 0.0f;
                pos[2] = -2.5f;
                pos[3] = 1.0;
                ixformPoint(pos, posw, modelView);
                posw[3] = 0.0f;

                psystem->addSphere(0, posw, velw, ballr, pr*2.0f);
            }
            break;

        case 'w':
            wireframe = !wireframe;
            break;

        case 'h':
            displaySliders = !displaySliders;
            break;
    }

    demoMode = false;
    idleCounter = 0;
    glutPostRedisplay();
}
void special(int k, int x, int y){
    if (displaySliders)
    {
        params->Special(k, x, y);
    }

    demoMode = false;
    idleCounter = 0;
}
void idle(void){
    if ((idleCounter++ > idleDelay) && (demoMode==false))
    {
        demoMode = true;
        printf("Entering demo mode\n");
    }

    if (demoMode)
    {
        camera_rot[1] += 0.1f;

        if (demoCounter++ > 1000)
        {
            ballr = 10 + (rand() % 10);
            addSphere();
            demoCounter = 0;
        }
    }

    glutPostRedisplay();
}
void initParams(){   
    params = new ParamListGL("misc");
    params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.01f, &timestep));
    params->AddParam(new Param<float>("damping"  , damping , 0.0f, 1.0f, 0.001f, &damping));
    params->AddParam(new Param<float>("gravity"  , gravity , 0.0f, 0.001f, 0.0001f, &gravity));
    params->AddParam(new Param<int> ("ball radius", ballr , 1, 20, 1, &ballr));
    params->AddParam(new Param<float>("collide spring" , collideSpring , 0.0f, 1.0f, 0.001f, &collideSpring));
    params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 0.1f, 0.001f, &collideDamping));
    params->AddParam(new Param<float>("collide shear"  , collideShear  , 0.0f, 0.1f, 0.001f, &collideShear));
    params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));
}
void mainMenu(int i){
    key((unsigned char) i, 0, 0);
}
void initMenus(){
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
    glutAddMenuEntry("Add sphere [3]", '3');
    glutAddMenuEntry("View mode [v]", 'v');
    glutAddMenuEntry("Move cursor mode [m]", 'm');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

int main(int argc, char **argv){
    //Local variables
	numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;
    numIterations = 0;	
    gridSize.x = gridSize.y = gridSize.z = gridDim;
	//Initialization
    initGL(&argc, argv);
    cudaInit(argc, argv);
    initParticleSystem(numParticles, gridSize, true);
    initParams();
    initMenus();
	CubeMapTexture();
	SmokeTexture();
	SkyBoxDataTransfer();
	//GLUT Functions
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutSpecialFunc(special);
    glutIdleFunc(idle);
    glutCloseFunc(cleanup);
    glutMainLoop();
}