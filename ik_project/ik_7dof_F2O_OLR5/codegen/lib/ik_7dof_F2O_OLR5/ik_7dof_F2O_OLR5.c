
#include "rt_nonfinite.h"
#include "ik_7dof_F2O_OLR5.h"
#include "norm.h"

/* Function Declarations */
static double rt_atan2d_snf(double u0, double u1);
static double rt_powd_snf(double u0, double u1);

static double rt_atan2d_snf(double u0, double u1)
{
  double y;
  int b_u0;
  int b_u1;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = rtNaN;
  } else if (rtIsInf(u0) && rtIsInf(u1)) {
    if (u0 > 0.0) {
      b_u0 = 1;
    } else {
      b_u0 = -1;
    }

    if (u1 > 0.0) {
      b_u1 = 1;
    } else {
      b_u1 = -1;
    }

    y = atan2(b_u0, b_u1);
  } else if (u1 == 0.0) {
    if (u0 > 0.0) {
      y = RT_PI / 2.0;
    } else if (u0 < 0.0) {
      y = -(RT_PI / 2.0);
    } else {
      y = 0.0;
    }
  } else {
    y = atan2(u0, u1);
  }

  return y;
}

/*
 * Arguments    : double u0
 *                double u1
 * Return Type  : double
 */
static double rt_powd_snf(double u0, double u1)
{
  double y;
  double d0;
  double d1;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = rtNaN;
  } else {
    d0 = fabs(u0);
    d1 = fabs(u1);
    if (rtIsInf(u1)) {
      if (d0 == 1.0) {
        y = rtNaN;
      } else if (d0 > 1.0) {
        if (u1 > 0.0) {
          y = rtInf;
        } else {
          y = 0.0;
        }
      } else if (u1 > 0.0) {
        y = 0.0;
      } else {
        y = rtInf;
      }
    } else if (d1 == 0.0) {
      y = 1.0;
    } else if (d1 == 1.0) {
      if (u1 > 0.0) {
        y = u0;
      } else {
        y = 1.0 / u0;
      }
    } else if (u1 == 2.0) {
      y = u0 * u0;
    } else if ((u1 == 0.5) && (u0 >= 0.0)) {
      y = sqrt(u0);
    } else if ((u0 < 0.0) && (u1 > floor(u1))) {
      y = rtNaN;
    } else {
      y = pow(u0, u1);
    }
  }

  return y;
}

/*
 * 20240520 Ӧ��3��ͻ��
 * Arguments    : double z_alpha
 *                double y_beta
 *                double x_gamma
 *                double p_x
 *                double p_y
 *                double p_z
 *                double bet
 *                const double cur_theta[7]
 *                double arm_LRflag
 *                double outTh[8]
 * Return Type  : void
 */
void ik_7dof_F2O_OLR5(double z_alpha, double y_beta, double x_gamma, double p_x,
                      double p_y, double p_z, double bet, const double
                      cur_theta[7], double arm_LRflag, double outTh[8]) 
                      //就用这个函数
                      
{
  double th_lim[14];
  int i;
  static const short iv0[14] = { -165, -105, -30, 0, -165, -105, -60, 165, 105,
    175, 165, 165, 105, 60 };

  int ik_state;
  double T07[16];
  double p[3];
  double L_BW;
  double theta4;
  double a;
  double d5;
  double n[3];
  double y;
  double p_e[3];
  double b_p_e[9];
  double Rk0x[3];
  double n4[3];
  int i0;
  double cB;
  double sB;
  double b_Rk0x[9];
  double c_Rk0x[9];
  double b_n;
  static const signed char iv1[3] = { 0, 0, 1 };

  int i1;
  double cond_BE;
  static const short b[3] = { 300, 0, 0 };

  double theta1;
  double dv0[16];
  double dv1[16];
  double dv2[16];
  static const signed char iv2[4] = { 0, 0, 1, 0 };

  double dv3[4];
  static const signed char iv3[4] = { 0, 0, 0, 1 };

  static const signed char b_b[4] = { 0, 0, 0, 1 };

  double theta3;
  double dv4[16];
  static const short iv4[4] = { 0, 0, -1, -300 };

  double theta6;
  double R47[9];
  double theta5;
  double theta7;
  double theta[7];
  double xx;
  double b_theta;
  for (i = 0; i < 14; i++) {
    th_lim[i] = iv0[i];
  }

  /*  if arm_LRflag==0 % L */
  if (arm_LRflag == 1.0) {
    /*  R */
    for (i = 0; i < 2; i++) {
      th_lim[2 + 7 * i] = -175.0 + 205.0 * (double)i;
    }
  }

  for (i = 0; i < 14; i++) {
    th_lim[i] = th_lim[i] * 3.1415926535897931 / 180.0;
  }

  ik_state = 0;

  /*  0��ʾ������1-7��ʾ����λ��11-17��ʾ����λ��25�����죬26�����죬27������ */
  /*  ��е�۳ߴ���� */
  /*  R_end = [cos(z_alpha)*cos(y_beta), cos(z_alpha)*sin(y_beta)*sin(x_gamma)-sin(z_alpha)*cos(x_gamma), cos(z_alpha)*sin(y_beta)*cos(x_gamma)+sin(z_alpha)*sin(x_gamma); */
  /*         sin(z_alpha)*cos(y_beta), sin(z_alpha)*sin(y_beta)*sin(x_gamma)+cos(z_alpha)*cos(x_gamma), sin(z_alpha)*sin(y_beta)*cos(x_gamma)-cos(z_alpha)*sin(x_gamma); */
  /*         -sin(y_beta), cos(y_beta)*sin(x_gamma), cos(y_beta)*cos(x_gamma)]; */
  T07[0] = cos(y_beta) * cos(z_alpha);
  T07[4] = -cos(y_beta) * sin(z_alpha);
  T07[8] = sin(y_beta);
  T07[1] = cos(x_gamma) * sin(z_alpha) + cos(z_alpha) * sin(x_gamma) * sin
    (y_beta);
  T07[5] = cos(x_gamma) * cos(z_alpha) - sin(x_gamma) * sin(y_beta) * sin
    (z_alpha);
  T07[9] = -cos(y_beta) * sin(x_gamma);
  T07[2] = sin(x_gamma) * sin(z_alpha) - cos(x_gamma) * cos(z_alpha) * sin
    (y_beta);
  T07[6] = cos(z_alpha) * sin(x_gamma) + cos(x_gamma) * sin(y_beta) * sin
    (z_alpha);
  T07[10] = cos(x_gamma) * cos(y_beta);
  T07[12] = p_x;
  T07[13] = p_y;
  T07[14] = p_z;
  T07[15] = 1.0;
  for (i = 0; i < 3; i++) {
    T07[3 + (i << 2)] = 0.0;
    p[i] = T07[12 + i];
  }

  for (i = 0; i < 2; i++) {
    p[i] = -T07[12 + i];
  }

  /* �ǵð���������귴һ�� */
  L_BW = norm(p);
  if (601.91693779125376 - L_BW > 1.0E-6) {
    /*  ����֮�ʹ��ڵ����ߣ���ʾ��û����ֱ */
    a = norm(p);
    theta4 = ((6.2831853071795862 - acos((181152.0 - a * a) / 181152.00000000003))
              - 1.4909663410826584) - 1.4909663410826584;
  } else {
    theta4 = cur_theta[3];
    ik_state = 25;

    /*  ������ؽ���ֱ */
  }

  if (ik_state == 25) {
    for (i = 0; i < 7; i++) {
      outTh[i] = cur_theta[i];
    }

    outTh[7] = 25.0;
  } else {
    theta4 = -theta4;

    /* �����ҿ������Ƕ��ǶԵģ���ô�ͱ���˸��Ƕ� */
    /* ���theta4�󣬾��ض���һ�»�е�� */
    d5 = sqrt((90000.0 + L_BW * L_BW) - 600.0 * L_BW * cos(0.0798299857122382 +
               acos(((90576.000000000015 + L_BW * L_BW) - 90576.000000000015) /
                    (2.0 * L_BW * 300.95846889562688))));

    /*  ��е�۳ߴ���� */
    n[0] = p[1] * -0.0 - p[2] * -0.0;
    n[1] = -p[2] - p[0] * -0.0;
    n[2] = p[0] * -0.0 - (-p[1]);

    /*  ��λ�淨����Ϊn */
    y = norm(n);
    a = norm(p);
    for (i = 0; i < 3; i++) {
      p_e[i] = p[i] / a;
      n[i] /= y;
    }

    /*  ��һ����p */
    bet += 3.1415926535897931;
    b_p_e[0] = p_e[0] * p_e[0] * (1.0 - cos(bet)) + cos(bet);
    b_p_e[3] = p_e[0] * p_e[1] * (1.0 - cos(bet)) - p_e[2] * sin(bet);
    b_p_e[6] = p_e[0] * p_e[2] * (1.0 - cos(bet)) + p_e[1] * sin(bet);
    b_p_e[1] = p_e[0] * p_e[1] * (1.0 - cos(bet)) + p_e[2] * sin(bet);
    b_p_e[4] = p_e[1] * p_e[1] * (1.0 - cos(bet)) + cos(bet);
    b_p_e[7] = p_e[1] * p_e[2] * (1.0 - cos(bet)) - p_e[0] * sin(bet);
    b_p_e[2] = p_e[0] * p_e[2] * (1.0 - cos(bet)) - p_e[1] * sin(bet);
    b_p_e[5] = p_e[1] * p_e[2] * (1.0 - cos(bet)) + p_e[0] * sin(bet);
    b_p_e[8] = p_e[2] * p_e[2] * (1.0 - cos(bet)) + cos(bet);
    for (i = 0; i < 3; i++) {
      n4[i] = 0.0;
      for (i0 = 0; i0 < 3; i0++) {
        n4[i] += b_p_e[i + 3 * i0] * n[i0];
      }
    }

    Rk0x[0] = p_e[0];
    Rk0x[1] = p_e[1];
    Rk0x[2] = p_e[2];
    p_e[0] = n4[0];
    p_e[1] = n4[1];
    p_e[2] = n4[2];
    n[0] = n4[1] * Rk0x[2] - n4[2] * Rk0x[1];
    n[1] = n4[2] * Rk0x[0] - n4[0] * Rk0x[2];
    n[2] = n4[0] * Rk0x[1] - n4[1] * Rk0x[0];

    /* �õ�k�᷽���y��0�е����� */
    y = norm(n);

    /* ��һ���������õ�k���y��0�е����� */
    a = norm(p);
    cB = ((90000.0 + a * a) - d5 * d5) / (600.0 * norm(p));
    sB = sqrt(1.0 - cB * cB);
    b_p_e[0] = cB;
    b_p_e[3] = -sB;
    b_p_e[6] = 0.0;
    b_p_e[1] = sB;
    b_p_e[4] = cB;
    b_p_e[7] = 0.0;
    for (i = 0; i < 3; i++) {
      b_n = n[i] / y;
      b_Rk0x[i] = Rk0x[i];
      b_Rk0x[3 + i] = b_n;
      b_Rk0x[6 + i] = p_e[i];
      b_p_e[2 + 3 * i] = iv1[i];
      n[i] = b_n;
    }

    for (i = 0; i < 3; i++) {
      n4[i] = 0.0;
      for (i0 = 0; i0 < 3; i0++) {
        c_Rk0x[i + 3 * i0] = 0.0;
        for (i1 = 0; i1 < 3; i1++) {
          c_Rk0x[i + 3 * i0] += b_Rk0x[i + 3 * i1] * b_p_e[i1 + 3 * i0];
        }

        n4[i] += c_Rk0x[i + 3 * i0] * (double)b[i0];
      }
    }

    /*  E��������0ϵ�µı�ʾ, xE=pE0(1), yE=pE0(2), zE=pE0(3) */
    /*  ��fai */
    a = acos(n4[2] / 300.0);

    /* �����漰��ѡ�ĸ��� */
    /*  �������theta2  */
    /*  theta2 = -theta2; */
    cond_BE = 0.0 * cos(a) + 300.0 * sin(a);
    if (fabs(cond_BE) > 1.0E-6) {
      theta1 = rt_atan2d_snf(n4[1] / cond_BE, n4[0] / cond_BE);
    } else {
      theta1 = cur_theta[0];
      ik_state = 26;

      /* ������ؽ����� */
    }

    /*  theta1 = pi - theta1; %��֪��������ʲô�����ﷴһ�¾ͺ��� */
    /*  �������theta1 */
    dv0[0] = cos(theta1);
    dv0[4] = -sin(theta1);
    dv0[8] = 0.0;
    dv0[12] = 0.0;
    dv0[1] = sin(theta1);
    dv0[5] = cos(theta1);
    dv0[9] = 0.0;
    dv0[13] = 0.0;
    dv1[0] = cos(a);
    dv1[4] = -sin(a);
    dv1[8] = 0.0;
    dv1[12] = 0.0;
    dv1[2] = -sin(a);
    dv1[6] = -cos(a);
    dv1[10] = 0.0;
    dv1[14] = 0.0;
    for (i = 0; i < 4; i++) {
      dv0[2 + (i << 2)] = iv2[i];
      dv0[3 + (i << 2)] = iv3[i];
      dv1[1 + (i << 2)] = iv2[i];
      dv1[3 + (i << 2)] = iv3[i];
    }

    for (i = 0; i < 4; i++) {
      dv3[i] = 0.0;
      for (i0 = 0; i0 < 4; i0++) {
        dv2[i + (i0 << 2)] = 0.0;
        for (i1 = 0; i1 < 4; i1++) {
          dv2[i + (i0 << 2)] += dv0[i + (i1 << 2)] * dv1[i1 + (i0 << 2)];
        }

        dv3[i] += dv2[i + (i0 << 2)] * (double)b_b[i0];
      }
    }

    for (i = 0; i < 3; i++) {
      p_e[i] = p[i] - dv3[i];
      p[i] -= n4[i];
    }

    n[0] = p_e[1] * p[2] - p_e[2] * p[1];
    n[1] = p_e[2] * p[0] - p_e[0] * p[2];
    n[2] = p_e[0] * p[1] - p_e[1] * p[0];
    y = norm(n);
    for (i = 0; i < 3; i++) {
      n[i] /= y;
    }

    if (sin(a) <= 1.0E-6) {
      theta3 = rt_atan2d_snf(-n[0], n[1]) - theta1;
    } else if ((fabs(sin(a)) > 1.0E-6) && (fabs(cos(theta1)) < 1.0E-6)) {
      theta3 = rt_atan2d_snf(n[2] / sin(a), -(n[0] + cos(theta1) * cos(a) * n[2]
        / sin(a)) / sin(theta1));
    } else {
      theta3 = rt_atan2d_snf(n[2] / sin(a), (n[1] + sin(theta1) * cos(a) * n[2] /
        sin(a)) / cos(theta1));
    }

    /*  �������theta3 */
    theta3 += 3.1415926535897931;

    /*       if arm_LRflag == 1 */
    if (theta3 > 3.1415926535897931) {
      theta3 -= 6.2831853071795862;
    }

    /*       end */
    theta4 = -theta4;
    dv0[0] = cos(theta1);
    dv0[4] = -sin(theta1);
    dv0[8] = 0.0;
    dv0[12] = 0.0;
    dv0[1] = sin(theta1);
    dv0[5] = cos(theta1);
    dv0[9] = 0.0;
    dv0[13] = 0.0;
    dv1[0] = cos(-a);
    dv1[4] = -sin(-a);
    dv1[8] = 0.0;
    dv1[12] = 0.0;
    dv1[2] = -sin(-a);
    dv1[6] = -cos(-a);
    dv1[10] = 0.0;
    dv1[14] = 0.0;
    for (i = 0; i < 4; i++) {
      dv0[2 + (i << 2)] = iv2[i];
      dv0[3 + (i << 2)] = iv3[i];
      dv1[1 + (i << 2)] = iv2[i];
      dv1[3 + (i << 2)] = iv3[i];
    }

    dv4[0] = cos(theta3);
    dv4[4] = -sin(theta3);
    dv4[8] = 0.0;
    dv4[12] = 0.0;
    dv4[2] = sin(theta3);
    dv4[6] = cos(theta3);
    dv4[10] = 0.0;
    dv4[14] = 0.0;
    for (i = 0; i < 4; i++) {
      for (i0 = 0; i0 < 4; i0++) {
        dv2[i + (i0 << 2)] = 0.0;
        for (i1 = 0; i1 < 4; i1++) {
          dv2[i + (i0 << 2)] += dv0[i + (i1 << 2)] * dv1[i1 + (i0 << 2)];
        }
      }

      dv4[1 + (i << 2)] = iv4[i];
      dv4[3 + (i << 2)] = iv3[i];
    }

    dv1[0] = cos(theta4);
    dv1[4] = -sin(theta4);
    dv1[8] = 0.0;
    dv1[12] = 0.0;
    dv1[2] = -sin(theta4);
    dv1[6] = -cos(theta4);
    dv1[10] = 0.0;
    dv1[14] = 0.0;
    for (i = 0; i < 4; i++) {
      for (i0 = 0; i0 < 4; i0++) {
        dv0[i + (i0 << 2)] = 0.0;
        for (i1 = 0; i1 < 4; i1++) {
          dv0[i + (i0 << 2)] += dv2[i + (i1 << 2)] * dv4[i1 + (i0 << 2)];
        }
      }

      dv1[1 + (i << 2)] = iv2[i];
      dv1[3 + (i << 2)] = iv3[i];
    }

    for (i = 0; i < 4; i++) {
      for (i0 = 0; i0 < 4; i0++) {
        dv2[i + (i0 << 2)] = 0.0;
        for (i1 = 0; i1 < 4; i1++) {
          dv2[i + (i0 << 2)] += dv0[i0 + (i1 << 2)] * dv1[i1 + (i << 2)];
        }
      }
    }

    for (i = 0; i < 3; i++) {
      for (i0 = 0; i0 < 3; i0++) {
        R47[i + 3 * i0] = 0.0;
        for (i1 = 0; i1 < 3; i1++) {
          R47[i + 3 * i0] += dv2[i + (i1 << 2)] * T07[i1 + (i0 << 2)];
        }
      }
    }

    theta6 = acos(-R47[7]);

    /* ע��������ѡ�����⣬���ұۿ��ܰɲ�ͬ */
    /*  �������theta6 */
    if (fabs(theta6) > 1.0E-6) {
      theta5 = rt_atan2d_snf(R47[8] / sin(theta6), R47[6] / sin(theta6));
      theta7 = rt_atan2d_snf(-R47[4] / sin(theta6), R47[1] / sin(theta6));
    } else {
      theta5 = cur_theta[4];

      /*  ��ʵ�����˸�����һʱ��ֵ */
      theta7 = rt_atan2d_snf(-R47[3], R47[5]) - cur_theta[4];
      ik_state = 27;

      /* ����ĩ�����죬һ�㲻�ᷢ�� */
    }

    theta6 -= 1.5707963267948966;

    /* ʱ����λƫ�� */
    theta[0] = theta1;
    theta[1] = -a;
    theta[2] = theta3;
    theta[3] = theta4;
    theta[4] = theta5;
    theta[5] = theta6;
    theta[6] = theta7;

    /*  7����λ���� */
    xx = fabs(theta6);
    th_lim[13] = (((-0.239248 * rt_powd_snf(xx, 4.0) + 1.22568 * rt_powd_snf(xx,
      3.0)) + -1.95134 * (xx * xx)) + 0.702285 * xx) + 0.861822;
    th_lim[6] = -th_lim[13];

    /*  if ik_state > 0 */
    /*       outTh = [cur_theta,ik_state]; */
    /*  else */
    for (i = 0; i < 7; i++) {
      b_theta = theta[i];

      /* һֱ��7���Ƿ�ӳ�ؽ���λ�� */
      if (theta[i] < th_lim[i]) {
        ik_state = 1 + i;
        b_theta = th_lim[i];
      } else {
        if (theta[i] > th_lim[7 + i]) {
          ik_state = 11 + i;
          b_theta = th_lim[7 + i];
        }
      }

      outTh[i] = b_theta;
    }

    outTh[7] = ik_state;

    /*  end */
  }

  /*  ����������end */
}

/*
 * File trailer for ik_7dof_F2O_OLR5.c
 *
 * [EOF]
 */

// double ik_arm(double z_alpha, double y_beta, double x_gamma, double p_x,
//                       double p_y, double p_z, double bet, const double
//                       cur_theta[7], double arm_LRflag)
//                       {
//                         double outTh[8];
//                         ik_7dof_F2O_OLR5(z_alpha, y_beta,  x_gamma,  p_x,\
//                       p_y,  p_z,  bet, cur_theta, arm_LRflag, outTh);
                      
//                       return outTh;
//                       }

// namespace py = pybind11;
// PYBIND11_MODULE(ik_module, m) {
//     m.def("ik_arm", &ik_arm, "Inverse Kinematics function",
//           py::arg("z_alpha"), py::arg("y_beta"), py::arg("x_gamma"),
//           py::arg("p_x"), py::arg("p_y"), py::arg("p_z"), py::arg("bet"),
//           py::arg("cur_theta"), py::arg("arm_LRflag"));
// }
