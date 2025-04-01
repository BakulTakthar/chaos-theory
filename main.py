from manim import *
import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def ode_solution_point(function, state0, time, dt=0.001):
    solution = solve_ivp(
        function,
        t_span=(0, time),
        y0=state0,
        t_eval=np.arange(0, time, dt),
        method="RK45"
    )
    return solution.y.T if solution.y.size > 0 else None

class LorenzAttractor(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=(-50, 50, 10),
            y_range=(-50, 50, 10),
            z_range=(0, 50, 10),
        ).set_color(WHITE)
        self.add(axes)

        # Solve the ODE
        state0 = [5, 5, 1]
        points = ode_solution_point(lorenz, state0, 10)
        
        label_x = axes.get_x_axis_label("x").set_color(WHITE)
        label_y = axes.get_y_axis_label("y").set_color(WHITE)
        label_z = axes.get_z_axis_label("z").set_color(WHITE)

        self.add(label_x, label_y, label_z)
        
        self.wait(5)
        
        self.camera.set_euler_angles(43 * DEGREES, 76 * DEGREES, 1 * DEGREES)
        self.camera.move_to(axes.get_center()-[0,0,0.7])
        # self.camera.set_distance(100)


        if points is None:
            print("Error: No points returned from ODE solver.")
            return

        epsilon = 0.001
        states = [ [0.5, 1 , 1.05 + n*epsilon] for n in range(2)]
        curves = VGroup()
        
        colors = [RED, BLUE]
        
        for state, color in zip(states, colors):
            points = ode_solution_point(lorenz, state, 10)
            if points is None:
                print("Error: No points returned from ODE solver.")
                return
            curve = ParametricFunction(
                lambda t: axes.c2p(*points[int(t * len(points) - 1)]),
                t_range=[0, 1, 0.01],  
                color=color
            )
            curves.add(curve)
            
        
        
        
        

        # # Create a 3D Parametric Function
        # curve = ParametricFunction(
        #     lambda t: axes.c2p(*filtered_points[int(t * len(filtered_points) - 1)]),
        #     t_range=[0, 1, 0.01],  
        #     color=BLUE
        # )

        
        self.play(*(Create(curve, run_time=20, rate_func=linear) for curve in curves))
        self.interactive_embed()
