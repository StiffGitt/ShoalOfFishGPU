#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib> 
#include <ctime> 
#include <Windows.h>
#include <algorithm>
#include <iterator>
#include "Shaders.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900 
#define ATTR_COUNT 5
#define N 500
#define SPECIES_COUNT 3
#define LEFT_EDGE -0.7f
#define RIGHT_EDGE 0.7f
#define BOTTOM_EDGE -0.7f
#define TOP_EDGE 0.7f
#define MARGIN 0.01f
#define TURN_COEF 0.00004
#define MINV 0.002
#define MAXV 0.01
#define RANGE1 0.01f
#define RANGE2 0.2f
#define CURSOR_RANGE 0.1f
#define CURSOR_COEF 0.1f
#define GET_R(type) (type == 0)? 1.0f : 0.0f;
#define GET_G(type) (type == 1)? 1.0f : 0.0f;
#define GET_B(type) (type == 2)? 1.0f : 0.0f;

struct Fish {
    float x;
    float dx;
    float y;
    float dy;
    int species;
};

struct Velocity {
    float x[N];
    float y[N];
};

class sort_indices
{
private:
    int* mparr;
public:
    sort_indices(int* parr) : mparr(parr) {}
    bool operator()(int i, int j) const { return mparr[i] < mparr[j]; }
};

GLFWwindow* window;
GLuint buffer, vao;
Fish fishes[N];
Fish gathered_fishes[N];

float A[3] = { 0.005f, 0.007f, 0.01f };
float H[3] = { 0.02f, 0.03f, 0.04f };
double cursorX = 0;
double cursorY = 0;
bool mouse_pressed = false;

int window_init();
void fishes_init();
float* get_vertices();
void buffer_init();
void draw_frame();
void move_fishes();
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);


void print_vertices(float* vertices);
void print_fish();

void assign_grid(float cell_size, int *cell_idx, int *indices)
{
    int grid_size = (int)(2.0f / cell_size) + 1;
    for (int i = 0; i < N; i++)
    {
        float x = fishes[i].x + 1.0f;
        float y = fishes[i].y + 1.0f;
        int r = (int)(y / cell_size);
        int c = (int)(x / cell_size);
        cell_idx[i] = r * grid_size + c;
    }
}

void find_border_cells(int *grid_first, int *grid_last, int *cell_idx)
{
    grid_first[cell_idx[0]] = 0;
    for (int i = 1; i < N; i++)
    {
        int cur_cell = cell_idx[i];
        int prev_cell = cell_idx[i - 1];
        if (cur_cell != prev_cell)
        {
            grid_last[prev_cell] = i;
            grid_first[cur_cell] = i;
        }
        if (i == N - 1)
            grid_last[cur_cell] = N;
    }
}

void gather(int *indices)
{
    for (int i = 0; i < N; i++)
    {
        gathered_fishes[i].x = fishes[indices[i]].x;
        gathered_fishes[i].y = fishes[indices[i]].y;
        gathered_fishes[i].dx = fishes[indices[i]].dx;
        gathered_fishes[i].dy = fishes[indices[i]].dy;
        gathered_fishes[i].species = fishes[indices[i]].species;
    }
}

void calculate_v(float r1, float r2, float turnCoef, float cohensionCoef, float avoidCoef, float alignCoef, float predatorsCoef,
    float preyCoef, float maxV, float minV, float curX, float curY, float curActive, bool predatorMode,
    int *grid_first, int *grid_last, int *cell_idx, int *indices)
{

    float cell_size = r2 * 2;
    int grid_size = (int)(2.0f / cell_size) + 1;
    int grid_length = (grid_size) * (grid_size);
    std::fill_n(grid_first, grid_length, -1);
    std::fill_n(grid_last, grid_length, -1);

    for (int i = 0; i < N; i++)
        indices[i] = i;
    assign_grid(cell_size, cell_idx, indices);

    std::sort(indices, indices + N, sort_indices(cell_idx));
    std::sort(cell_idx, cell_idx + N);

    find_border_cells(grid_first, grid_last, cell_idx);

    gather(indices);

    Velocity vel;
    float r1sq = r1 * r1;
    float r2sq = r2 * r2;

    for (int i = 0; i < N; i++)
    {
        float x = gathered_fishes[i].x;
        float y = gathered_fishes[i].y;
        float vx = gathered_fishes[i].dx;
        float vy = gathered_fishes[i].dy;
        float cumX = 0.0, cumY = 0.0, cumVx = 0.0, cumVy = 0.0, visibleFriendlyCount = 0.0, visiblePreyCount = 0.0,
            closestPredatorX = -1.0, closestPredatorY = -1.0f, closeDx = 0.0, closeDy = 0.0, cumXP = 0.0, cumYP = 0.0,
            closestPredatorDsq = 8.0f;
        int cell = cell_idx[i];
        int cells_to_check[] = { cell - 1, cell, cell + 1,
            cell - grid_length - 1, cell - grid_length, cell - grid_length + 1,
            cell - grid_length + 1, cell + grid_length, cell + grid_length + 1 };
        for (int idx = 0; idx <= 8; idx++)
        {
            int nc = cells_to_check[idx];
            if (nc < 0 || nc > grid_length || grid_first[nc] < 0)
                continue;
            for (int j = grid_first[nc]; j < grid_last[nc]; j++)
            {
                if (j == i)
                    continue;
                float xj = gathered_fishes[j].x;
                float yj = gathered_fishes[j].y;
                float dx = x - xj;
                float dy = y - yj;

                if (fabsf(dx) < r2 && fabsf(dy) < r2)
                {
                    float dsq = dx * dx + dy * dy;
                    if (dsq < r2)
                    {
                        // Avoid predators
                        if (gathered_fishes[i].species < gathered_fishes[j].species)
                        {
                            if (closestPredatorDsq > dsq)
                            {
                                closestPredatorDsq = dsq;
                                closestPredatorX = xj;
                                closestPredatorY = yj;
                            }
                        }
                        // Hunt prey
                        if (gathered_fishes[i].species > gathered_fishes[j].species)
                        {
                            visiblePreyCount++;
                            cumXP += xj;
                            cumYP += yj;
                        }
                        if (dsq < r1sq)
                        {
                            // Separation
                            closeDx += (x - xj); /** (1 - (dx / r1));*/
                            closeDy += (y - yj); /** (1 - (dy / r1));*/
                        }
                        else
                        {
                            if (gathered_fishes[i].species == gathered_fishes[j].species && gathered_fishes[i].species <= 1)
                            {
                                visibleFriendlyCount++;
                                // Alignment
                                cumVx += gathered_fishes[j].dx;
                                cumVy += gathered_fishes[j].dy;

                                // Cohension
                                cumX += xj;
                                cumY += yj;
                            }
                        }
                    }
                }
            }
        }
        
        // Avoid predators
        if (predatorMode && closestPredatorDsq < r2)
        {
            vx += (x - closestPredatorX) * predatorsCoef;
            vy += (y - closestPredatorY) * predatorsCoef;
        }

        // Chase prey
        if (predatorMode && visiblePreyCount > 0)
        {
            vx += ((cumXP / visiblePreyCount) - x) * preyCoef;
            vy += ((cumYP / visiblePreyCount) - y) * preyCoef;
        }

        // Separation
        vx += closeDx * avoidCoef;
        vy += closeDy * avoidCoef;

        if (visibleFriendlyCount > 0)
        {
            // Alignment
            vx += ((cumVx / visibleFriendlyCount) - gathered_fishes[i].dx) * alignCoef;
            vy += ((cumVy / visibleFriendlyCount) - gathered_fishes[i].dy) * alignCoef;

            // Cohension
            vx += ((cumX / visibleFriendlyCount) - x) * cohensionCoef;
            vy += ((cumY / visibleFriendlyCount) - y) * cohensionCoef;
        }



        // Turn from edges
        bool isTurning = false;
        if (x < LEFT_EDGE && vx < minV)
        {
            isTurning = true;
            if (x < -1.0 + MARGIN)
                vx = -vx;
            else
                vx += turnCoef + (vx * vx) / (x + 1.0f);
        }
        if (x > RIGHT_EDGE && vx > -minV)
        {
            isTurning = true;
            if (x > 1.0 - MARGIN)
                vx = -vx;
            else
                vx -= turnCoef + (vx * vx) / (1.0f - x);
        }

        if (y < BOTTOM_EDGE && vy < minV)
        {
            isTurning = true;
            if (y < -1.0 + MARGIN)
                vy = -vy;
            else
                vy += turnCoef + (vy * vy) / (y + 1.0f);
        }
        if (y > TOP_EDGE && vy > -minV)
        {
            isTurning = true;
            if (y > 1.0 - MARGIN)
                vy = -vy;
            else
               vy -= turnCoef + (vy * vy) / (1.0f - y);
        }

        float dcx = x - curX;
        float dcy = y - curY;
        if (curActive &&  dcx * dcx + dcy * dcy < CURSOR_RANGE * CURSOR_RANGE)
        {
            vx += dcx * CURSOR_COEF;
            vy += dcy * CURSOR_COEF;
        }

        // Adjust velocity to min:max
        float v = sqrtf(vx * vx + vy * vy);
        if (v < minV && !isTurning)
        {
            vx = (vx / v) * minV;
            vy = (vy / v) * minV;
        }
        else if (v > maxV)
        {
            vx = (vx / v) * maxV;
            vy = (vy / v) * maxV;
        }

        vel.x[i] = vx;
        vel.y[i] = vy;
    }

    for (int i = 0; i < N; i++)
    {
        fishes[indices[i]].dx = vel.x[i];
        fishes[indices[i]].dy = vel.y[i];
    }

    
}

int main(void)
{
    float r1 = RANGE1, cohensionCoef = 0.25, avoidCoef = 0.5, alignCoef = 0.5, predatorsCoef = 0.5f, preyCoef = 0.3f;
    float turnCoef = TURN_COEF;
    if (window_init())
        return -1;

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    fishes_init();
    float cell_size = RANGE2 * 2;
    int grid_length = ((int)(2.0f / cell_size) + 1) * ((int)(2.0f / cell_size) + 1);
    int* grid_first = (int*)malloc(grid_length * sizeof(int));
    int* grid_last = (int*)malloc(grid_length * sizeof(int));
    int* cell_idx = (int*)malloc(N * sizeof(int));
    int* indices = (int*)malloc(N * sizeof(int));
    buffer_init();

    GLuint shader = StartShaders("res/shaders/Basic.shader");
     
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    bool shouldPause = false, predatorMode = false;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        draw_frame();

        // Imgui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Configuration");
        ImGui::SliderFloat("Separation range", &r1, 0.0f, RANGE2);
        ImGui::SliderFloat("Separation", &avoidCoef, 0.0f, 1.0f);
        ImGui::SliderFloat("Cohension", &cohensionCoef, 0.0f, 1.0f);
        ImGui::SliderFloat("Alignment", &alignCoef, 0.0f, 1.0f);
        ImGui::Checkbox("Predator mode", &predatorMode);
        ImGui::SliderFloat("Avoid predators", &predatorsCoef, 0.0f, 1.0f);
        ImGui::SliderFloat("Chase prey", &preyCoef, 0.0f, 1.0f);
        if (ImGui::Button("Pause"))
            shouldPause = !shouldPause;

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        //Sleep(20);
        if (!shouldPause)
        {
            calculate_v(r1, RANGE2, turnCoef, cohensionCoef / 1000.0f, avoidCoef / 100.0f, alignCoef / 100.0f, predatorsCoef / 50.0f,
                preyCoef / 100.0f, MAXV, MINV, ((cursorX / WINDOW_WIDTH) * 2) - 1, ((cursorY / WINDOW_HEIGHT) * 2) - 1, mouse_pressed, 
                predatorMode, grid_first, grid_last, cell_idx, indices);
            move_fishes();
        }
    }

    glDeleteProgram(shader);

    free(grid_first);
    free(grid_last);
    free(cell_idx);
    free(indices);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}

int window_init()
{
    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Shoal Of Fish", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK)
        std::cout << "glewInit ERROR" << std::endl;

    std::cout << glGetString(GL_VERSION) << std::endl;

    return 0;
}

void fishes_init()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < N; i++)
    {
        fishes[i].x = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX)) * 2.0f - 1.0f;
        fishes[i].y = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX)) * 2.0f - 1.0f;
        fishes[i].dx = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX) / 50.0f) * ( (std::rand() % 2)? -1.0f : 1.0f);
        fishes[i].dy = (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX) / 50.0f) * ( (std::rand() % 2) ? -1.0f : 1.0f);
        if (i % 50 == 0)
            fishes[i].species = 2;
        else if (i % 10 == 0)
            fishes[i].species = 1;
        else
            fishes[i].species = 0;
    }
}

float* get_vertices()
{
    float* vertices = (float*)malloc(N * 3 * ATTR_COUNT * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        float x = fishes[i].x;
        float y = fishes[i].y;
        float dx = fishes[i].dx;
        float dy = fishes[i].dy;
        int species = fishes[i].species;
        float d = sqrtf(dx * dx + dy * dy);

        vertices[i * 3 * ATTR_COUNT] = x - A[species] * (dy / d);
        vertices[i * 3 * ATTR_COUNT + 1] = y + A[species] * (dx / d);
        vertices[i * 3 * ATTR_COUNT + 2] = GET_R(species);
        vertices[i * 3 * ATTR_COUNT + 3] = GET_G(species);
        vertices[i * 3 * ATTR_COUNT + 4] = GET_B(species);

        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT] = x + A[species] * (dy / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 1] = y - A[species] * (dx / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 2] = GET_R(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 3] = GET_G(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT + 4] = GET_B(species);

        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2] = x + H[species] * (dx / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 1] = y + H[species] * (dy / d);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 2] = GET_R(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 3] = GET_G(species);
        vertices[i * 3 * ATTR_COUNT + ATTR_COUNT * 2 + 4] = GET_B(species);
    }
    return vertices;
}

void buffer_init()
{
    float* vertices = get_vertices();

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * ATTR_COUNT * N * sizeof(float), vertices, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, ATTR_COUNT * sizeof(float), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, ATTR_COUNT * sizeof(float), (void*)(sizeof(float) * 2));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void draw_frame()
{
    float* vertices = get_vertices();
    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * ATTR_COUNT * N * sizeof(float), vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDrawArrays(GL_TRIANGLES, 0, 3 * N);

    glBindVertexArray(0);
}

void move_fishes()
{
    for (int i = 0; i < N; i++)
    {
        fishes[i].x += fishes[i].dx;
        fishes[i].y += fishes[i].dy;
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (mouse_pressed)
    {
        cursorX = xpos;
        cursorY = WINDOW_HEIGHT - ypos;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        cursorX = xpos;
        cursorY = WINDOW_HEIGHT - ypos;
        mouse_pressed = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        mouse_pressed = false;
    }
}

void print_vertices(float* vertices)
{
    for (int i = 0; i < 3 * N; i++)
    {
        //if (vertices[i * ATTR_COUNT] > 1.0f)
        //    vertices[i * ATTR_COUNT] = 1.0f;
        //if (vertices[i * ATTR_COUNT] < 0.0f)
        //    vertices[i * ATTR_COUNT] = 0.0f;
        //if (vertices[i * ATTR_COUNT + 1] > 1.0f)
        //    vertices[i * ATTR_COUNT + 1] = 1.0f;
        //if (vertices[i * ATTR_COUNT + 1] < 0.0f)
        //    vertices[i * ATTR_COUNT + 1] = 0.0f;
        std::cout << "x1 = " << vertices[i  * ATTR_COUNT] << std::endl;
        std::cout << "y1 = " << vertices[i  * ATTR_COUNT + 1] << std::endl;
        std::cout << "r = " << vertices[i * ATTR_COUNT + 2] << std::endl;
        std::cout << "g = " << vertices[i * ATTR_COUNT + 3] << std::endl;
        std::cout << "b = " << vertices[i * ATTR_COUNT + 4] << std::endl;

        std::cout << "---------------------" << std::endl;
    }

    /*for (int i = 0; i < 3 * N; i++)
    {
        std::cout << vertices[i * ATTR_COUNT] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 1] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 2] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 3] << " , ";
        std::cout << vertices[i * ATTR_COUNT + 4] << " , " << std::endl;
    }*/
}

void print_fish()
{
    for (int i = 0; i < N; i++)
    {
        std::cout << "vx = " << fishes[i].dx << std::endl;
        std::cout << "vy = " << fishes[i].dy << std::endl;
    }
}