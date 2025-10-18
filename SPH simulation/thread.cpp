



#include <GLFW/glfw3.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <thread>

int num_threads = 14;

const double PI = 3.14159265358979323846;
const float strength = 40000;

const int widthInitial = 1000;
const int heightInitial = 1000;
int width = widthInitial;
int height = heightInitial;

float viscocity = 2.0f;

const int num_particles = 1000;
const float mass = 1.0f;
const float smoothing_radius = 16.f;
const float gravity = 0.0f;
const float timeStep = 0.005f;
const int stepsPerRender = 1;
const float max_speed = 200.0f;

const float cell_size = smoothing_radius;
std::vector<std::vector<int>> gridCells;
int gridCols, gridRows;

const float rest_density = 8.0f;
const float gas_constant = 200.0f;
const float damping = 0.6f;

struct Particle {
    float x, y;
    float vx = 0.f, vy = 0.f;
    float size;
    float density = 0.f;
    Particle(float nx, float ny, float ns) : x(nx), y(ny), size(ns) {}
};

std::vector<Particle> particles;

double mouseX = 0.0, mouseY = 0.0;
bool mousePressed = false;

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    mouseX = xpos;
    mouseY = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) mousePressed = true;
        if (action == GLFW_RELEASE) mousePressed = false;
    }
}

inline int cellIndex(int cx, int cy) {
    return cy * gridCols + cx;
}

inline std::pair<int,int> worldToCell(float x, float y) {
    int cx = std::clamp((int)(x / cell_size), 0, gridCols - 1);
    int cy = std::clamp((int)(y / cell_size), 0, gridRows - 1);
    return {cx, cy};
}

const float C = 4.0/(PI * std::pow(smoothing_radius, 8));
const float C6 = -6*C;
const float hh = smoothing_radius*smoothing_radius;
static inline float poly6_W(float r) {
    if (r >= smoothing_radius) return 0.f;
    float val = hh - r*r;
    return C * val * val * val;
}

const float visc_const = (45.0f / (PI * std::pow(smoothing_radius, 6)));

static inline float viscosity_laplacian(float r) {
    if (r >= smoothing_radius) return 0.f;
    return visc_const * (smoothing_radius - r);
}

static inline std::pair<float,float> poly6_gradW_vec(float dx, float dy, float r) {
    if (r >= smoothing_radius || r == 0.f) return {0.f, 0.f};
    float val = hh - r*r;
    float scale = C6 * val * val;
    return { scale * dx, scale * dy };
}

void init_particles() {
    particles.clear();
    particles.reserve(num_particles);
    std::srand((unsigned)std::time(nullptr));

    int cols = (int)std::sqrt(num_particles);
    int rows = (num_particles + cols - 1) / cols;
    float spacing = smoothing_radius * 0.5f;

    float box_width  = cols * spacing;
    float box_height = rows * spacing;

    bool fits_on_screen = (box_width <= width && box_height <= height);

    int created = 0;
    if (fits_on_screen) {
        float startX = (width  - box_width)  / 2.0f;
        float startY = (height - box_height) / 2.0f;

        for (int r = 0; r < rows && created < num_particles; ++r) {
            for (int c = 0; c < cols && created < num_particles; ++c) {
                float x = startX + c * spacing + (std::rand() % 10 - 5);
                float y = startY + r * spacing + (std::rand() % 10 - 5);
                particles.emplace_back(x, y, 5.0f);
                ++created;
            }
        }
    } else {
        for (int i = 0; i < num_particles; ++i) {
            float x = static_cast<float>(std::rand() % width);
            float y = static_cast<float>(std::rand() % height);
            particles.emplace_back(x, y, 3.0f);
        }
    }
}

// Parallel helper
template<typename Func>
void parallel_for(int num_threads, int n, Func func) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int chunk = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end   = std::min(start + chunk, n);
        if (start >= end) break;
        threads.emplace_back([=, &func]() {
            for (int i = start; i < end; i++) {
                func(i);
            }
        });
    }
    for (auto &th : threads) th.join();
}

void update_densities() {
    parallel_for(num_threads, (int)particles.size(), [&](int i) {
        auto [cx, cy] = worldToCell(particles[i].x, particles[i].y);
        float dens = 0.f;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int ncx = cx + dx;
                int ncy = cy + dy;
                if (ncx < 0 || ncy < 0 || ncx >= gridCols || ncy >= gridRows) continue;

                for (int j : gridCells[cellIndex(ncx, ncy)]) {
                    float dxp = particles[j].x - particles[i].x;
                    float dyp = particles[j].y - particles[i].y;
                    float r = std::hypot(dxp, dyp);
                    if (r < smoothing_radius) {
                        dens += mass * poly6_W(r);
                    }
                }
            }
        }
        particles[i].density = std::max(dens, 1e-6f);
    });
}

inline float density_to_pressure(float density) {
    return gas_constant * (density - rest_density);
}

void compute_pressure_forces_and_apply(float dt) {
    parallel_for(num_threads, (int)particles.size(), [&](int i) {
        float Pi = density_to_pressure(particles[i].density);
        float rhoi = particles[i].density;

        float fx = 0.f, fy = 0.f;
        float visc_x = 0.0f, visc_y = 0.0f;

        auto [cx, cy] = worldToCell(particles[i].x, particles[i].y);

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int ncx = cx + dx;
                int ncy = cy + dy;
                if (ncx < 0 || ncy < 0 || ncx >= gridCols || ncy >= gridRows) continue;

                for (int j : gridCells[cellIndex(ncx, ncy)]) {
                    if (j == i) continue;

                    float dxp = particles[j].x - particles[i].x;
                    float dyp = particles[j].y - particles[i].y;
                    float r = std::hypot(dxp, dyp);
                    if (r >= smoothing_radius || r == 0.f) continue;

                    float Pj = density_to_pressure(particles[j].density);
                    float rhoj = particles[j].density;

                    auto g = poly6_gradW_vec(dxp, dyp, r);

                    float coeff = - mass * (Pi + Pj) / (2.0f * rhoj);
                    fx += coeff * g.first;
                    fy += coeff * g.second;

                    float visc = viscosity_laplacian(r);
                    visc_x += (particles[j].vx - particles[i].vx) * visc / rhoj;
                    visc_y += (particles[j].vy - particles[i].vy) * visc / rhoj;
                }
            }
        }

        fx += viscocity * mass * visc_x;
        fy += viscocity * mass * visc_y;

        float ax = fx / rhoi;
        float ay = fy / rhoi;

        particles[i].vx += ax * dt;
        particles[i].vy += ay * dt;
    });
}

void handle_boundaries(Particle &p) {
    if (p.x - p.size < 0.f) {
        p.x = p.size;
        p.vx = -p.vx * damping;
    }
    if (p.x + p.size > width) {
        p.x = width - p.size;
        p.vx = -p.vx * damping;
    }
    if (p.y - p.size < 0.f) {
        p.y = p.size;
        p.vy = -p.vy * damping;
    }
    if (p.y + p.size > height) {
        p.y = height - p.size;
        p.vy = -p.vy * damping;
    }
}

void neighbourhood_search_update() {
    gridCols = (int)(width / cell_size) + 1;
    gridRows = (int)(height / cell_size) + 1;
    gridCells.clear();
    gridCells.resize(gridCols * gridRows);

    for (int i = 0; i < (int)particles.size(); i++) {
        auto [cx, cy] = worldToCell(particles[i].x, particles[i].y);
        gridCells[cellIndex(cx, cy)].push_back(i);
    }
}

void drawCircle(float cx, float cy, float r, int num_segments) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(cx, cy);
    for (int i = 0; i <= num_segments; i++) {
        float theta = 2.0f * PI * float(i) / float(num_segments);
        float x = r * cosf(theta);
        float y = r * sinf(theta);
        glVertex2f(x + cx, y + cy);
    }
    glEnd();
}

int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "SPH Fluid (multithreaded)", NULL, NULL);
    if (!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    init_particles();

    long frames = 0;

    while (!glfwWindowShouldClose(window)) {
        for (int s = 0; s < stepsPerRender; ++s) {
            neighbourhood_search_update();
            update_densities();
            compute_pressure_forces_and_apply(timeStep);

            for (auto &p : particles) {
                p.vy += gravity * timeStep;

                p.vx = std::max(std::min(p.vx, max_speed), -max_speed);
                p.vy = std::max(std::min(p.vy, max_speed), -max_speed);

                p.x += p.vx * timeStep;
                p.y += p.vy * timeStep;

                handle_boundaries(p);

                if (mousePressed) {
                    float dx = (float)mouseX - p.x;
                    float dy = (float)mouseY - p.y;
                    float dist = std::hypot(dx, dy) + 1e-5f;

                    float force = strength / dist;

                    p.vx += (dx / dist) * force * timeStep;
                    p.vy += (dy / dist) * force * timeStep;
                }
            }
        }

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        for (auto &p : particles) {
            float val = (fabs(p.vx)+fabs(p.vy))/max_speed;
            glColor3f(val, 0.8f, 1-val);
            drawCircle(p.x, p.y, p.size, 10);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        frames++;
        std::cout << "Frame: " << frames 
                  << "  time: " << frames*stepsPerRender*timeStep 
                  << "  particles: " << particles.size() << "\n";
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
