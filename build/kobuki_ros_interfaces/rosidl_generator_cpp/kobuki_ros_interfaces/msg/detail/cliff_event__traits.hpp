// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from kobuki_ros_interfaces:msg/CliffEvent.idl
// generated code does not contain a copyright notice

#ifndef KOBUKI_ROS_INTERFACES__MSG__DETAIL__CLIFF_EVENT__TRAITS_HPP_
#define KOBUKI_ROS_INTERFACES__MSG__DETAIL__CLIFF_EVENT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "kobuki_ros_interfaces/msg/detail/cliff_event__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace kobuki_ros_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const CliffEvent & msg,
  std::ostream & out)
{
  out << "{";
  // member: sensor
  {
    out << "sensor: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor, out);
    out << ", ";
  }

  // member: state
  {
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
    out << ", ";
  }

  // member: bottom
  {
    out << "bottom: ";
    rosidl_generator_traits::value_to_yaml(msg.bottom, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CliffEvent & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: sensor
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sensor: ";
    rosidl_generator_traits::value_to_yaml(msg.sensor, out);
    out << "\n";
  }

  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
    out << "\n";
  }

  // member: bottom
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bottom: ";
    rosidl_generator_traits::value_to_yaml(msg.bottom, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CliffEvent & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace kobuki_ros_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use kobuki_ros_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const kobuki_ros_interfaces::msg::CliffEvent & msg,
  std::ostream & out, size_t indentation = 0)
{
  kobuki_ros_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use kobuki_ros_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const kobuki_ros_interfaces::msg::CliffEvent & msg)
{
  return kobuki_ros_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<kobuki_ros_interfaces::msg::CliffEvent>()
{
  return "kobuki_ros_interfaces::msg::CliffEvent";
}

template<>
inline const char * name<kobuki_ros_interfaces::msg::CliffEvent>()
{
  return "kobuki_ros_interfaces/msg/CliffEvent";
}

template<>
struct has_fixed_size<kobuki_ros_interfaces::msg::CliffEvent>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<kobuki_ros_interfaces::msg::CliffEvent>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<kobuki_ros_interfaces::msg::CliffEvent>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // KOBUKI_ROS_INTERFACES__MSG__DETAIL__CLIFF_EVENT__TRAITS_HPP_
