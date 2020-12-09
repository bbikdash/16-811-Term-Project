"""
@author: Bassam Bikdash, Ivan Cisneros
Interpolation.py

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""
import numpy as np
from scipy.interpolate import CubicSpline

class Interpolation:
    @staticmethod
    def cubic_spline_scipy(indices, data, numpoints=6):
        """
        By default, uses the Not-a-knot type of interpolation.
        
        @param numPoints must be an even number
        """
        x = np.zeros(numpoints, dtype=int)

        # Use points around the clipped region to create the interpolated spline
        mid = int(numpoints/2)
        for i in range(0, mid):
            x[i] = indices[0] - (mid-i-1)
            x[numpoints - (1 + i)] = indices[1] + (mid-i-1)
        y = data[x]

        return CubicSpline(x, y)


    @staticmethod
    def cubic_spline(indices, data, numpoints=6):
        """
        A from scratch implementation of the Cubic Spline Interpolation.
        """
        
        # TODO: Complete
        """
        function [S C]=Spline3(u,x,y,dya,dyb)
            % vectors x and y contain n+1 points and the corresponding function values
            % vector u contains all discrete samples of the continuous argument of f(x)
            % dya and dyb are the derivatives f'(x_0) and f'(x_n), respectively 
            n=length(x);       % number of interpolating points
            k=length(u);       % number of discrete sample points
            C=zeros(n,k);      % the n-1 cubic interpolating polynomials
            A=2*eye(n);        % coefficient matrix on left-hand side
            A(1,2)=1;
            A(n,n-1)=1;   
            d=zeros(n,1);      % vector on right-hand side
            d(1)=((y(2)-y(1))/(x(2)-x(1))-dya)/h0;  % first element of d
            for i=2:n-1
                h0=x(i)-x(i-1);
                h1=x(i+1)-x(i);
                h2=x(i+1)-x(i-1);       
                A(i,i-1)=h0/h2;
                A(i,i+1)=h1/h2;
                d(i)=((y(i+1)-y(i))/h1-(y(i)-y(i-1))/h0)/h2; % 2nd divided difference
            end
            d(n)=(dyb-(y(n)-y(n-1))/h1)/h1;   % last element of d
            M=6*inv(A)*d;                     % solving linear equation system for M's
            for i=2:n
                h=x(i)-x(i-1);
                x0=u-x(i-1);
                x1=x(i)-u;
                C(i-1,:)=(x1.^3*M(i-1)+x0.^3*M(i))/6/h... % the ith cubic polynomial
                         -(M(i-1)*x1+M(i)*x0)*h/6+(y(i-1)*x1+y(i)*x0)/h;  
                idx=find(u>x(i-1) & u<=x(i));  % indices between x(i-1) and x(i)
                S(idx)=C(i-1,idx);             % constructing spline by cubic polynomials
            end
        end
        """
        # TODO
        pass

