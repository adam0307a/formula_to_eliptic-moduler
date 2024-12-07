import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button
import cmath

class EllipticCurve:
    def __init__(self, a, b):
        """
        Initialize an elliptic curve of the form y^2 = x^3 + ax + b
        """
        self.a = a
        self.b = b

    def get_y_values(self, x):
        """
        Calculate y values for a given x value
        """
        y_squared = x**3 + self.a * x + self.b
        # Only return real values
        if y_squared >= 0:
            return [np.sqrt(y_squared), -np.sqrt(y_squared)]
        return []

    def plot_curve(self):
        """
        Plot the elliptic curve
        """
        x = np.linspace(-5, 5, 1000)
        points_x = []
        points_y = []
        
        for xi in x:
            y_values = self.get_y_values(xi)
            for y in y_values:
                points_x.append(xi)
                points_y.append(y)

        plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 2)
        
        # Plot elliptic curve
        ax1 = plt.subplot(gs[0])
        ax1.plot(points_x, points_y, 'b-', label=f'y² = x³ + {self.a}x + {self.b}')
        ax1.grid(True)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Elliptic Curve: y² = x³ + {self.a}x + {self.b}')
        ax1.legend()
        ax1.axis('equal')
        
        # Plot modular form representation (simplified)
        ax2 = plt.subplot(gs[1])
        t = np.linspace(0, 2*np.pi, 1000)
        r = np.abs(np.sin(t) + np.cos(t))  # Simplified modular form representation
        ax2.plot(r * np.cos(t), r * np.sin(t), 'r-', label='Modular Form Representation')
        ax2.grid(True)
        ax2.set_xlabel('Re')
        ax2.set_ylabel('Im')
        ax2.set_title('Modular Form')
        ax2.legend()
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()

    def discriminant(self):
        """
        Calculate the discriminant of the elliptic curve: Δ = -16(4a³ + 27b²)
        """
        return -16 * (4 * self.a**3 + 27 * self.b**2)

    def j_invariant(self, tau=None):
        """
        Calculate j-invariant. For Weierstrass form:
        j = -1728(4a³)/(4a³ + 27b²) = -1728(4a³)/Δ
        
        If tau is provided, calculates j-invariant using q-expansion:
        j(τ) = 1/q + 744 + 196884q + 21493760q² + ...
        where q = exp(2πiτ)
        """
        if tau is not None:
            # q-expansion method
            q = np.exp(2 * np.pi * 1j * tau)
            # Using first few terms of the series
            j = (1/q + 744 + 196884*q + 21493760*q**2)
            return j
        else:
            # Weierstrass form method
            discriminant = self.discriminant()
            if discriminant == 0:
                return float('inf')
            j = -1728 * (4 * self.a**3) / discriminant
            return j

    def get_elliptic_curve_params(self, tau):
        """
        Given a point tau in the upper half-plane,
        return the corresponding elliptic curve parameters a and b
        """
        q = np.exp(2 * np.pi * 1j * tau)
        # Eisenstein series E4 and E6
        E4 = 1 + 240*q + 2160*q**2
        E6 = 1 - 504*q - 16632*q**2
        
        # Convert to Weierstrass form
        a = -27 * E4 / (4 * E6)
        b = -27 * (E4**3 - E6**2) / (4 * E6**2)
        
        return float(a.real), float(b.real)

    def plot_modular_form(self, ax):
        """
        Display upper half-plane representation for modular form
        """
        # Create grid for upper half-plane
        x = np.linspace(-3, 3, 40)
        y = np.linspace(0.1, 3, 40)
        X, Y = np.meshgrid(x, y)
        tau = X + 1j*Y  # Complex grid points
        
        # Calculate j-invariant for each point
        Z = np.zeros_like(X, dtype=complex)
        for i in range(len(x)):
            for j in range(len(y)):
                current_tau = tau[j,i]
                # Get corresponding elliptic curve parameters
                a_tau, b_tau = self.get_elliptic_curve_params(current_tau)
                # Calculate j-invariant for these parameters
                discriminant = -16 * (4 * a_tau**3 + 27 * b_tau**2)
                if discriminant != 0:
                    Z[j,i] = -1728 * (4 * a_tau**3) / discriminant
                else:
                    Z[j,i] = float('inf')
        
        # Create color map based on j-invariant values
        Z_abs = np.log(np.abs(Z) + 1)  # Magnitude
        
        # Plot magnitude as contour
        contour = ax.contourf(X, Y, Z_abs, levels=30, cmap='viridis')
        colorbar = plt.colorbar(contour, ax=ax)
        colorbar.set_label('log(|j-invariant| + 1)')
        
        # Draw fundamental domain boundaries
        ax.plot([-1, -0.5], [0, np.sqrt(3)/2], 'r--', alpha=0.5, linewidth=2, label='Fundamental Domain')
        ax.plot([-0.5, 0.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'r--', alpha=0.5, linewidth=2)
        ax.plot([0.5, 1], [np.sqrt(3)/2, 0], 'r--', alpha=0.5, linewidth=2)
        
        # Draw additional copies of fundamental domain
        for k in range(-2, 3):
            if k != 0:
                ax.plot([k-1, k-0.5], [0, np.sqrt(3)/2], 'r:', alpha=0.3)
                ax.plot([k-0.5, k+0.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'r:', alpha=0.3)
                ax.plot([k+0.5, k+1], [np.sqrt(3)/2, 0], 'r:', alpha=0.3)
        
        # Add special points
        i_point = (0, 1)
        rho_point = (0.5, np.sqrt(3)/2)
        
        # Calculate j-invariants for special points
        j_i = self.j_invariant(i_point[0] + 1j*i_point[1])
        j_rho = self.j_invariant(rho_point[0] + 1j*rho_point[1])
        
        # Plot special points with annotations
        ax.plot(i_point[0], i_point[1], 'ro', label=f'i (j={j_i.real:.0f})', markersize=8, markeredgecolor='white')
        ax.plot(rho_point[0], rho_point[1], 'bo', label=f'ρ (j={j_rho.real:.0f})', markersize=8, markeredgecolor='white')
        
        # Add annotations with better positioning
        ax.annotate('4-fold\nrotation', xy=i_point, xytext=(-0.5, 1.5),
                   arrowprops=dict(facecolor='red', shrink=0.05), fontsize=8)
        ax.annotate('3-fold\nrotation', xy=rho_point, xytext=(1.0, 1.5),
                   arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=8)
        
        ax.set_xlabel('Re(τ)')
        ax.set_ylabel('Im(τ)')
        ax.set_title(f'Modular Form in Upper Half-Plane\nj-invariant = {self.j_invariant():.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(0.1, 3)
        ax.legend()
        
        return colorbar

    def plot_current_curve(self, ax):
        """
        Helper function to plot the current curve on given axes
        """
        x = np.linspace(-5, 5, 1000)
        points_x = []
        points_y = []
        
        for xi in x:
            y_values = self.get_y_values(xi)
            for y in y_values:
                points_x.append(xi)
                points_y.append(y)
        
        ax.plot(points_x, points_y, 'b-', label=f'y² = x³ + {self.a}x + {self.b}')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Elliptic Curve: y² = x³ + {self.a}x + {self.b}')
        ax.legend()
        ax.axis('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    def interactive_plot(self):
        """
        Interactive plot with text boxes for a and b parameters
        """
        # Create main figure and subplots
        self.fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1, 1.2])
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        
        plt.subplots_adjust(bottom=0.2)
        
        # Initial plots
        self.plot_current_curve(self.ax1)
        self.current_colorbar = self.plot_modular_form(self.ax2)
        
        # Create textboxes and button
        axbox_a = plt.axes([0.2, 0.05, 0.1, 0.05])
        axbox_b = plt.axes([0.5, 0.05, 0.1, 0.05])
        text_box_a = TextBox(axbox_a, 'a = ', initial=str(self.a))
        text_box_b = TextBox(axbox_b, 'b = ', initial=str(self.b))
        axbutton = plt.axes([0.7, 0.05, 0.1, 0.05])
        button = Button(axbutton, 'Update')
        
        def update_plot(event=None):
            try:
                # Get new values
                self.a = float(text_box_a.text)
                self.b = float(text_box_b.text)
                
                # Remove old colorbar
                if hasattr(self, 'current_colorbar'):
                    self.current_colorbar.remove()
                
                # Clear plots
                self.ax1.clear()
                self.ax2.clear()
                
                # Redraw plots
                self.plot_current_curve(self.ax1)
                self.current_colorbar = self.plot_modular_form(self.ax2)
                
                # Update figure
                plt.draw()
            except ValueError:
                print("Please enter valid numbers")
        
        # Bind click event to update button
        button.on_clicked(update_plot)
        
        # Bind enter key events to textboxes
        text_box_a.on_submit(update_plot)
        text_box_b.on_submit(update_plot)
        
        plt.show()

def main():
    curve = EllipticCurve(a=-1, b=0)
    curve.interactive_plot()

if __name__ == "__main__":
    main()
