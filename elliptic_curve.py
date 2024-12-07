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
        ax1.set_title('Elliptic Curve')
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
        """
        discriminant = self.discriminant()
        if discriminant == 0:
            return float('inf')
        
        j = -1728 * (4 * self.a**3) / discriminant
        return j

    def plot_modular_form(self, ax):
        """
        Display upper half-plane representation for modular form
        """
        # Calculate j-invariant value
        j = self.j_invariant()
        
        # Create grid for upper half-plane
        x = np.linspace(-2, 2, 20)
        y = np.linspace(0.1, 2, 20)
        X, Y = np.meshgrid(x, y)
        
        # Create color map based on j-invariant values
        Z = np.full_like(X, np.abs(j))
        Z = np.log(Z + 1)  # Logarithmic scale
        
        # Contour plot
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        colorbar = plt.colorbar(contour, ax=ax)
        
        # Draw fundamental domain boundaries
        ax.plot([-1, -0.5], [0, np.sqrt(3)/2], 'r--', alpha=0.5)
        ax.plot([-0.5, 0.5], [np.sqrt(3)/2, np.sqrt(3)/2], 'r--', alpha=0.5)
        ax.plot([0.5, 1], [np.sqrt(3)/2, 0], 'r--', alpha=0.5)
        
        ax.set_xlabel('Re(τ)')
        ax.set_ylabel('Im(τ)')
        ax.set_title(f'Modular Form (j = {j:.2f})')
        ax.grid(True)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0.1, 2)
        
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
        ax.set_title('Elliptic Curve')
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
